using CrackSegmentationApp.Models;
using CrackSegmentationApp.Utilities;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Media.Imaging;

namespace CrackSegmentationApp.Services;

/// <summary>
/// Handles post-processing of ONNX model outputs
/// Implements softmax, argmax, and centerline extraction
/// </summary>
public class ImagePostprocessor
{
    /// <summary>
    /// Processes the ONNX model output to generate visualization images
    /// </summary>
    /// <param name="logits">Raw logits from ONNX model [2, H, W]</param>
    /// <param name="height">Padded image height</param>
    /// <param name="width">Padded image width</param>
    /// <param name="origHeight">Original image height (before padding)</param>
    /// <param name="origWidth">Original image width (before padding)</param>
    /// <param name="originalImage">Original input image for result</param>
    /// <returns>Segmentation result with all output images</returns>
    public SegmentationResult ProcessOutput(float[] logits, int height, int width, int origHeight, int origWidth, Bitmap originalImage)
    {
        // Step 1: Compute softmax probabilities from logits
        var softmax = ComputeSoftmax(logits, height, width);

        // Step 2: Compute argmax (binary segmentation mask)
        var argmax = ComputeArgmax(softmax, height, width);

        // Step 3: Extract crack class (channel 1) from softmax
        var crackProbability = ExtractChannel(softmax, 1, height, width);

        // Step 4: Crop to original dimensions (remove padding)
        crackProbability = CropFloatArray(crackProbability, origHeight, origWidth);
        argmax = CropByteArray(argmax, origHeight, origWidth);

        // Step 5: Compute centerlines using morphological thinning
        var centerlines = MorphologyOperations.MorphologicalThinning(argmax);

        // Step 6: Convert to images for display (invert for visualization to match Python)
        var result = new SegmentationResult
        {
            OriginalImage = Utilities.ImageConverter.BitmapToBitmapSource(originalImage),
            SoftmaxImage = ConvertToGrayscaleImage(InvertFloatValues(crackProbability), origHeight, origWidth),
            ArgmaxImage = ConvertToGrayscaleImage(InvertByteValues(argmax), origHeight, origWidth),
            CenterlinesImage = ConvertToGrayscaleImage(InvertByteValues(centerlines), origHeight, origWidth),
            ImageSize = (origWidth, origHeight)
        };

        return result;
    }

    /// <summary>
    /// Crops a float array to the specified dimensions (removes padding)
    /// </summary>
    private float[,] CropFloatArray(float[,] data, int targetHeight, int targetWidth)
    {
        float[,] cropped = new float[targetHeight, targetWidth];

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                cropped[y, x] = data[y, x];
            }
        }

        return cropped;
    }

    /// <summary>
    /// Crops a byte array to the specified dimensions (removes padding)
    /// </summary>
    private byte[,] CropByteArray(byte[,] data, int targetHeight, int targetWidth)
    {
        byte[,] cropped = new byte[targetHeight, targetWidth];

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                cropped[y, x] = data[y, x];
            }
        }

        return cropped;
    }

    /// <summary>
    /// Computes softmax: exp(x_i) / sum(exp(x_j))
    /// Converts logits to probabilities
    /// </summary>
    private float[,] ComputeSoftmax(float[] logits, int height, int width)
    {
        int numClasses = 2;  // background and crack
        float[,] softmax = new float[numClasses * height * width, 1];
        int pixelCount = height * width;

        // Apply softmax for each pixel
        for (int i = 0; i < pixelCount; i++)
        {
            // Get logits for both classes at this pixel
            float logit0 = logits[0 * pixelCount + i];  // background
            float logit1 = logits[1 * pixelCount + i];  // crack

            // Compute exp for numerical stability (subtract max)
            float maxLogit = Math.Max(logit0, logit1);
            float exp0 = (float)Math.Exp(logit0 - maxLogit);
            float exp1 = (float)Math.Exp(logit1 - maxLogit);
            float sumExp = exp0 + exp1;

            // Store probabilities
            softmax[0 * pixelCount + i, 0] = exp0 / sumExp;  // background probability
            softmax[1 * pixelCount + i, 0] = exp1 / sumExp;  // crack probability
        }

        return softmax;
    }

    /// <summary>
    /// Computes argmax - finds the class with highest probability for each pixel
    /// </summary>
    private byte[,] ComputeArgmax(float[,] softmax, int height, int width)
    {
        byte[,] argmax = new byte[height, width];
        int pixelCount = height * width;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pixelIndex = y * width + x;

                float prob0 = softmax[0 * pixelCount + pixelIndex, 0];  // background
                float prob1 = softmax[1 * pixelCount + pixelIndex, 0];  // crack

                // Assign class label (0 or 1), multiply by 255 for visualization
                argmax[y, x] = (byte)(prob1 > prob0 ? 255 : 0);
            }
        }

        return argmax;
    }

    /// <summary>
    /// Extracts a specific channel from the softmax output
    /// </summary>
    private float[,] ExtractChannel(float[,] softmax, int channel, int height, int width)
    {
        float[,] extracted = new float[height, width];
        int pixelCount = height * width;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pixelIndex = y * width + x;
                extracted[y, x] = softmax[channel * pixelCount + pixelIndex, 0];
            }
        }

        return extracted;
    }

    /// <summary>
    /// Inverts float values: 1.0 - value
    /// Python: softmax = 1 - softmax
    /// </summary>
    private float[,] InvertFloatValues(float[,] data)
    {
        int height = data.GetLength(0);
        int width = data.GetLength(1);
        float[,] inverted = new float[height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                inverted[y, x] = 1.0f - data[y, x];
            }
        }

        return inverted;
    }

    /// <summary>
    /// Inverts byte values: 255 - value
    /// Python: argmax = 255 - argmax
    /// </summary>
    private byte[,] InvertByteValues(byte[,] data)
    {
        int height = data.GetLength(0);
        int width = data.GetLength(1);
        byte[,] inverted = new byte[height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                inverted[y, x] = (byte)(255 - data[y, x]);
            }
        }

        return inverted;
    }

    /// <summary>
    /// Converts a float array [0.0-1.0] to a grayscale BitmapSource
    /// </summary>
    private BitmapSource ConvertToGrayscaleImage(float[,] data, int height, int width)
    {
        Bitmap bitmap = new Bitmap(width, height, PixelFormat.Format8bppIndexed);

        // Set grayscale palette
        ColorPalette palette = bitmap.Palette;
        for (int i = 0; i < 256; i++)
        {
            palette.Entries[i] = Color.FromArgb(i, i, i);
        }
        bitmap.Palette = palette;

        // Lock bitmap for writing
        BitmapData bmpData = bitmap.LockBits(
            new Rectangle(0, 0, width, height),
            ImageLockMode.WriteOnly,
            PixelFormat.Format8bppIndexed);

        unsafe
        {
            byte* ptr = (byte*)bmpData.Scan0;
            int stride = bmpData.Stride;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Convert float [0.0-1.0] to byte [0-255]
                    byte value = (byte)(Math.Clamp(data[y, x], 0.0f, 1.0f) * 255);
                    ptr[y * stride + x] = value;
                }
            }
        }

        bitmap.UnlockBits(bmpData);

        return Utilities.ImageConverter.BitmapToBitmapSource(bitmap);
    }

    /// <summary>
    /// Converts a byte array to a grayscale BitmapSource
    /// </summary>
    private BitmapSource ConvertToGrayscaleImage(byte[,] data, int height, int width)
    {
        Bitmap bitmap = new Bitmap(width, height, PixelFormat.Format8bppIndexed);

        // Set grayscale palette
        ColorPalette palette = bitmap.Palette;
        for (int i = 0; i < 256; i++)
        {
            palette.Entries[i] = Color.FromArgb(i, i, i);
        }
        bitmap.Palette = palette;

        // Lock bitmap for writing
        BitmapData bmpData = bitmap.LockBits(
            new Rectangle(0, 0, width, height),
            ImageLockMode.WriteOnly,
            PixelFormat.Format8bppIndexed);

        unsafe
        {
            byte* ptr = (byte*)bmpData.Scan0;
            int stride = bmpData.Stride;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    ptr[y * stride + x] = data[y, x];
                }
            }
        }

        bitmap.UnlockBits(bmpData);

        return Utilities.ImageConverter.BitmapToBitmapSource(bitmap);
    }
}
