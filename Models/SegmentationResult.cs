using System.Windows.Media.Imaging;

namespace CrackSegmentationApp.Models;

/// <summary>
/// Represents the result of crack segmentation inference
/// </summary>
public class SegmentationResult
{
    /// <summary>
    /// Original input image
    /// </summary>
    public BitmapSource? OriginalImage { get; set; }

    /// <summary>
    /// Softmax probability map for crack class (inverted for visualization)
    /// </summary>
    public BitmapSource? SoftmaxImage { get; set; }

    /// <summary>
    /// Binary segmentation mask (argmax, inverted for visualization)
    /// </summary>
    public BitmapSource? ArgmaxImage { get; set; }

    /// <summary>
    /// Centerlines/skeleton of detected cracks (morphological thinning)
    /// </summary>
    public BitmapSource? CenterlinesImage { get; set; }

    /// <summary>
    /// Inference time in milliseconds
    /// </summary>
    public double InferenceTimeMs { get; set; }

    /// <summary>
    /// Image dimensions
    /// </summary>
    public (int Width, int Height) ImageSize { get; set; }
}
