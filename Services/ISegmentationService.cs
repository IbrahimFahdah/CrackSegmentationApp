using CrackSegmentationApp.Models;
using System.Drawing;

namespace CrackSegmentationApp.Services;

/// <summary>
/// Interface for crack segmentation service
/// </summary>
public interface ISegmentationService
{
    /// <summary>
    /// Performs crack segmentation on the input image
    /// </summary>
    /// <param name="inputImage">Input image to segment</param>
    /// <returns>Segmentation results including softmax, argmax, and centerlines</returns>
    SegmentationResult PerformSegmentation(Bitmap inputImage);
}
