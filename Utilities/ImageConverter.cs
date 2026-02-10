using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Windows.Media.Imaging;

namespace CrackSegmentationApp.Utilities;

/// <summary>
/// Utility class for converting between Bitmap and BitmapSource
/// </summary>
public static class ImageConverter
{
    /// <summary>
    /// Converts System.Drawing.Bitmap to System.Windows.Media.Imaging.BitmapSource
    /// </summary>
    public static BitmapSource BitmapToBitmapSource(Bitmap bitmap)
    {
        using var memory = new MemoryStream();
        bitmap.Save(memory, ImageFormat.Png);
        memory.Position = 0;

        var bitmapImage = new BitmapImage();
        bitmapImage.BeginInit();
        bitmapImage.StreamSource = memory;
        bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
        bitmapImage.EndInit();
        bitmapImage.Freeze();

        return bitmapImage;
    }

    /// <summary>
    /// Converts System.Windows.Media.Imaging.BitmapSource to System.Drawing.Bitmap
    /// </summary>
    public static Bitmap BitmapSourceToBitmap(BitmapSource bitmapSource)
    {
        using var outStream = new MemoryStream();

        BitmapEncoder encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmapSource));
        encoder.Save(outStream);
        outStream.Position = 0;

        return new Bitmap(outStream);
    }

    /// <summary>
    /// Loads an image file and converts it to Bitmap
    /// </summary>
    public static Bitmap LoadImage(string filePath)
    {
        // Load as BitmapImage first to handle various formats
        var bitmapImage = new BitmapImage();
        bitmapImage.BeginInit();
        bitmapImage.UriSource = new Uri(filePath, UriKind.Absolute);
        bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
        bitmapImage.EndInit();

        return BitmapSourceToBitmap(bitmapImage);
    }
}
