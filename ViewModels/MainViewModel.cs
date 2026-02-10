using CrackSegmentationApp.Services;
using CrackSegmentationApp.Utilities;
using Microsoft.Win32;
using System.IO;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace CrackSegmentationApp.ViewModels;

/// <summary>
/// Main ViewModel for the crack segmentation application
/// </summary>
public class MainViewModel : ViewModelBase
{
    private readonly ISegmentationService _segmentationService;
    private BitmapSource? _originalImage;
    private BitmapSource? _softmaxImage;
    private BitmapSource? _argmaxImage;
    private BitmapSource? _centerlinesImage;
    private string _statusMessage = "Ready";
    private bool _isProcessing;
    private string? _currentImagePath;

    public MainViewModel()
    {
        // Initialize service with ONNX model path
        var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "crack_segmentation_fold0.onnx");
        _segmentationService = new OnnxSegmentationService(modelPath);

        // Initialize commands
        LoadImageCommand = new RelayCommand(LoadImage);
        RunSegmentationCommand = new RelayCommand(RunSegmentation, () => IsImageLoaded && !IsProcessing);
        SaveResultsCommand = new RelayCommand(SaveResults, () => HasResults && !IsProcessing);
    }

    #region Properties

    public BitmapSource? OriginalImage
    {
        get => _originalImage;
        set
        {
            if (SetProperty(ref _originalImage, value))
            {
                OnPropertyChanged(nameof(IsImageLoaded));
                ((RelayCommand)RunSegmentationCommand).RaiseCanExecuteChanged();
            }
        }
    }

    public BitmapSource? SoftmaxImage
    {
        get => _softmaxImage;
        set
        {
            if (SetProperty(ref _softmaxImage, value))
            {
                OnPropertyChanged(nameof(HasResults));
                ((RelayCommand)SaveResultsCommand).RaiseCanExecuteChanged();
            }
        }
    }

    public BitmapSource? ArgmaxImage
    {
        get => _argmaxImage;
        set => SetProperty(ref _argmaxImage, value);
    }

    public BitmapSource? CenterlinesImage
    {
        get => _centerlinesImage;
        set => SetProperty(ref _centerlinesImage, value);
    }

    public string StatusMessage
    {
        get => _statusMessage;
        set => SetProperty(ref _statusMessage, value);
    }

    public bool IsProcessing
    {
        get => _isProcessing;
        set
        {
            if (SetProperty(ref _isProcessing, value))
            {
                ((RelayCommand)RunSegmentationCommand).RaiseCanExecuteChanged();
                ((RelayCommand)SaveResultsCommand).RaiseCanExecuteChanged();
            }
        }
    }

    public bool IsImageLoaded => OriginalImage != null;
    public bool HasResults => SoftmaxImage != null;

    #endregion

    #region Commands

    public ICommand LoadImageCommand { get; }
    public ICommand RunSegmentationCommand { get; }
    public ICommand SaveResultsCommand { get; }

    #endregion

    #region Command Implementations

    private void LoadImage()
    {
        var openFileDialog = new OpenFileDialog
        {
            Filter = "Image files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp|All files (*.*)|*.*",
            Title = "Select an image for crack segmentation"
        };

        if (openFileDialog.ShowDialog() == true)
        {
            try
            {
                _currentImagePath = openFileDialog.FileName;

                // Load image as BitmapSource
                var bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.UriSource = new Uri(_currentImagePath, UriKind.Absolute);
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
                bitmapImage.Freeze();

                OriginalImage = bitmapImage;

                // Clear previous results
                SoftmaxImage = null;
                ArgmaxImage = null;
                CenterlinesImage = null;

                StatusMessage = $"Loaded: {Path.GetFileName(_currentImagePath)} ({bitmapImage.PixelWidth}x{bitmapImage.PixelHeight})";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error loading image: {ex.Message}";
                MessageBox.Show($"Failed to load image: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }

    private async void RunSegmentation()
    {
        if (OriginalImage == null || _currentImagePath == null)
            return;

        IsProcessing = true;
        StatusMessage = "Running segmentation...";

        try
        {
            // Load image as Bitmap for processing
            var bitmap = ImageConverter.LoadImage(_currentImagePath);

            // Run segmentation on background thread
            var result = await Task.Run(() => _segmentationService.PerformSegmentation(bitmap));

            // Update UI with results on UI thread
            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                SoftmaxImage = result.SoftmaxImage;
                ArgmaxImage = result.ArgmaxImage;
                CenterlinesImage = result.CenterlinesImage;

                StatusMessage = $"Segmentation complete! ({result.InferenceTimeMs:F0} ms)";
            });
        }
        catch (InvalidOperationException ex)
        {
            // Model not loaded error
            StatusMessage = "Error: ONNX model not found";
            MessageBox.Show(ex.Message, "Model Not Found", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
            MessageBox.Show($"Segmentation failed:\n\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            IsProcessing = false;
        }
    }

    private void SaveResults()
    {
        if (SoftmaxImage == null || ArgmaxImage == null || CenterlinesImage == null || _currentImagePath == null)
            return;

        try
        {
            // Create output directory
            var inputFileName = Path.GetFileNameWithoutExtension(_currentImagePath);
            var outputDir = Path.Combine(Path.GetDirectoryName(_currentImagePath) ?? "", $"{inputFileName}_results");
            Directory.CreateDirectory(outputDir);

            // Save each result image
            SaveBitmapSource(SoftmaxImage, Path.Combine(outputDir, "softmax.png"));
            SaveBitmapSource(ArgmaxImage, Path.Combine(outputDir, "argmax.png"));
            SaveBitmapSource(CenterlinesImage, Path.Combine(outputDir, "centerlines.png"));

            StatusMessage = $"Results saved to: {outputDir}";
            MessageBox.Show($"Results saved successfully to:\n{outputDir}", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error saving results: {ex.Message}";
            MessageBox.Show($"Failed to save results:\n\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void SaveBitmapSource(BitmapSource image, string filePath)
    {
        using var fileStream = new FileStream(filePath, FileMode.Create);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(image));
        encoder.Save(fileStream);
    }

    #endregion
}
