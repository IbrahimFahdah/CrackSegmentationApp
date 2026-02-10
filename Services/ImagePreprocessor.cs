using System.Drawing;
using System.Drawing.Imaging;

namespace CrackSegmentationApp.Services;

/// <summary>
/// Handles image preprocessing for ONNX inference
/// Implements the same preprocessing logic as the Python implementation
/// </summary>
public class ImagePreprocessor
{
    private readonly Random _random = new();

    /// <summary>
    /// Preprocesses the image for ONNX inference
    /// </summary>
    /// <param name="image">Input bitmap image</param>
    /// <param name="height">Output height of the image (padded to multiple of 64)</param>
    /// <param name="width">Output width of the image (padded to multiple of 64)</param>
    /// <param name="origHeight">Original image height before padding</param>
    /// <param name="origWidth">Original image width before padding</param>
    /// <returns>Flattened float array in [1, 3, H, W] format for ONNX</returns>
    public float[] PreprocessImage(Bitmap image, out int height, out int width, out int origHeight, out int origWidth)
    {
        // Calculate padded dimensions (must be multiples of 64 for nnU-Net)
        origHeight = image.Height;
        origWidth = image.Width;
        height = RoundUpToMultiple(origHeight, 64);
        width = RoundUpToMultiple(origWidth, 64);

        // Step 1: Convert bitmap to float array and pad to required dimensions
        var data = BitmapToFloatArrayWithPadding(image, height, width);

        // Step 2: Handle zero patches - replace all-zero pixels with small random noise
        // This prevents NaN during normalization
        ReplaceZeroPatches(data, height, width);

        // Step 3: Calculate per-channel mean and std
        var (means, stds) = ComputeImageStatistics(data, height, width);

        // Step 4: Apply Z-score normalization: (pixel - mean) / std
        NormalizeChannels(data, means, stds, height, width);

        return data;
    }

    /// <summary>
    /// Rounds up a value to the nearest multiple of the specified factor
    /// </summary>
    private int RoundUpToMultiple(int value, int multiple)
    {
        return ((value + multiple - 1) / multiple) * multiple;
    }

    /// <summary>
    /// Converts a Bitmap to float array in [C, H, W] layout with padding to target dimensions
    /// </summary>
    private float[] BitmapToFloatArrayWithPadding(Bitmap image, int targetHeight, int targetWidth)
    {
        int origWidth = image.Width;
        int origHeight = image.Height;
        float[] data = new float[3 * targetHeight * targetWidth];

        // Initialize with zeros (padding)
        Array.Fill(data, 0f);

        // Lock the bitmap for fast pixel access
        BitmapData bmpData = image.LockBits(
            new Rectangle(0, 0, origWidth, origHeight),
            ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);

        unsafe
        {
            byte* ptr = (byte*)bmpData.Scan0;
            int stride = bmpData.Stride;

            // Convert BGR to RGB and store in [C, H, W] format (top-left aligned with padding on right and bottom)
            for (int y = 0; y < origHeight; y++)
            {
                for (int x = 0; x < origWidth; x++)
                {
                    int pixelIndex = y * stride + x * 3;
                    byte b = ptr[pixelIndex];
                    byte g = ptr[pixelIndex + 1];
                    byte r = ptr[pixelIndex + 2];

                    // Store in [C, H, W] format using padded dimensions
                    int baseOffset = y * targetWidth + x;
                    data[0 * targetHeight * targetWidth + baseOffset] = r;  // R channel
                    data[1 * targetHeight * targetWidth + baseOffset] = g;  // G channel
                    data[2 * targetHeight * targetWidth + baseOffset] = b;  // B channel
                }
            }
        }

        image.UnlockBits(bmpData);
        return data;
    }

    /// <summary>
    /// Replaces all-zero pixels with small random noise to prevent NaN during normalization
    /// Python equivalent: img[:, torch.all(img == 0, dim=0)] = torch.rand_like(img)[:, torch.all(img == 0, dim=0)]
    /// </summary>
    private void ReplaceZeroPatches(float[] data, int height, int width)
    {
        int pixelCount = height * width;

        for (int i = 0; i < pixelCount; i++)
        {
            int rIndex = 0 * pixelCount + i;
            int gIndex = 1 * pixelCount + i;
            int bIndex = 2 * pixelCount + i;

            // Check if all channels are zero for this pixel
            if (data[rIndex] == 0 && data[gIndex] == 0 && data[bIndex] == 0)
            {
                // Replace with small random values [0, 1]
                data[rIndex] = (float)_random.NextDouble();
                data[gIndex] = (float)_random.NextDouble();
                data[bIndex] = (float)_random.NextDouble();
            }
        }
    }

    /// <summary>
    /// Computes per-channel mean and standard deviation
    /// </summary>
    private (float[] means, float[] stds) ComputeImageStatistics(float[] data, int height, int width)
    {
        float[] means = new float[3];
        float[] stds = new float[3];
        int pixelCount = height * width;

        // Calculate mean for each channel
        for (int c = 0; c < 3; c++)
        {
            double sum = 0;
            int channelOffset = c * pixelCount;

            for (int i = 0; i < pixelCount; i++)
            {
                sum += data[channelOffset + i];
            }

            means[c] = (float)(sum / pixelCount);
        }

        // Calculate standard deviation for each channel
        for (int c = 0; c < 3; c++)
        {
            double sumSquaredDiff = 0;
            int channelOffset = c * pixelCount;

            for (int i = 0; i < pixelCount; i++)
            {
                double diff = data[channelOffset + i] - means[c];
                sumSquaredDiff += diff * diff;
            }

            stds[c] = (float)Math.Sqrt(sumSquaredDiff / pixelCount);

            // Prevent division by zero - use 1.0 if std is very small
            if (stds[c] < 1e-6f)
            {
                stds[c] = 1.0f;
            }
        }

        return (means, stds);
    }

    /// <summary>
    /// Applies Z-score normalization: (pixel - mean) / std
    /// </summary>
    private void NormalizeChannels(float[] data, float[] means, float[] stds, int height, int width)
    {
        int pixelCount = height * width;

        for (int c = 0; c < 3; c++)
        {
            int channelOffset = c * pixelCount;
            float mean = means[c];
            float std = stds[c];

            for (int i = 0; i < pixelCount; i++)
            {
                data[channelOffset + i] = (data[channelOffset + i] - mean) / std;
            }
        }
    }
}
