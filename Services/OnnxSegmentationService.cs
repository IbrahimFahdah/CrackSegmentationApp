using CrackSegmentationApp.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using System.Drawing;
using System.IO;

namespace CrackSegmentationApp.Services;

/// <summary>
/// ONNX-based crack segmentation service
/// Implements the complete inference pipeline: preprocessing -> ONNX inference -> postprocessing
/// </summary>
public class OnnxSegmentationService : ISegmentationService, IDisposable
{
    private readonly InferenceSession? _session;
    private readonly ImagePreprocessor _preprocessor;
    private readonly ImagePostprocessor _postprocessor;
    private readonly string? _modelPath;
    private readonly bool _modelLoaded;

    public OnnxSegmentationService(string modelPath)
    {
        _modelPath = modelPath;
        _preprocessor = new ImagePreprocessor();
        _postprocessor = new ImagePostprocessor();

        // Check if model file exists
        if (!File.Exists(modelPath))
        {
            _modelLoaded = false;
            Console.WriteLine($"Warning: ONNX model not found at {modelPath}");
            return;
        }

        try
        {
            // Create ONNX Runtime session
            var sessionOptions = new SessionOptions();

            // Use all available CPU cores
            sessionOptions.IntraOpNumThreads = Environment.ProcessorCount;
            sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;

            _session = new InferenceSession(modelPath, sessionOptions);
            _modelLoaded = true;

            // Log model information
            Console.WriteLine($"ONNX model loaded successfully from {modelPath}");
            Console.WriteLine($"Input: {_session.InputMetadata.First().Key}");
            Console.WriteLine($"Output: {_session.OutputMetadata.First().Key}");
        }
        catch (Exception ex)
        {
            _modelLoaded = false;
            Console.WriteLine($"Failed to load ONNX model: {ex.Message}");
            _session = null;
        }
    }

    /// <summary>
    /// Performs crack segmentation on the input image
    /// </summary>
    public SegmentationResult PerformSegmentation(Bitmap inputImage)
    {
        // Check if model is loaded
        if (!_modelLoaded || _session == null)
        {
            throw new InvalidOperationException(
                $"ONNX model not loaded. Please ensure the model file exists at: {_modelPath}\n\n" +
                "To export the model:\n" +
                "1. Install Python with required packages (torch, onnx, onnxruntime, nnunetv2)\n" +
                "2. Run: python export_to_onnx.py\n" +
                "3. The model will be exported to CrackSegmentationApp/Models/crack_segmentation_fold0.onnx");
        }

        var stopwatch = Stopwatch.StartNew();

        try
        {
            // Step 1: Preprocess image (with padding to multiple of 64)
            var inputData = _preprocessor.PreprocessImage(inputImage, out int paddedHeight, out int paddedWidth,
                                                          out int origHeight, out int origWidth);

            // Step 2: Create ONNX tensor [1, 3, H, W]
            var dimensions = new int[] { 1, 3, paddedHeight, paddedWidth };
            var inputTensor = new DenseTensor<float>(inputData, dimensions);

            // Step 3: Create input for ONNX Runtime
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_session.InputMetadata.First().Key, inputTensor)
            };

            // Step 4: Run inference
            using var results = _session.Run(inputs);

            // Step 5: Extract output logits
            var outputTensor = results.First().AsTensor<float>();
            var outputData = outputTensor.ToArray();

            // Step 6: Post-process to generate visualizations (crop back to original size)
            var result = _postprocessor.ProcessOutput(outputData, paddedHeight, paddedWidth,
                                                      origHeight, origWidth, inputImage);

            stopwatch.Stop();
            result.InferenceTimeMs = stopwatch.Elapsed.TotalMilliseconds;

            return result;
        }
        catch (Exception ex)
        {
            throw new Exception($"Segmentation failed: {ex.Message}", ex);
        }
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
