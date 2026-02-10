namespace CrackSegmentationApp.Utilities;

/// <summary>
/// Morphological operations for image processing
/// Implements Zhang-Suen thinning algorithm to match Python's skimage.morphology.thin
/// </summary>
public static class MorphologyOperations
{
    /// <summary>
    /// Performs morphological thinning (skeletonization) on a binary image
    /// Uses the Zhang-Suen thinning algorithm
    /// </summary>
    /// <param name="image">Binary image (0 or 255)</param>
    /// <returns>Thinned/skeletonized image</returns>
    public static byte[,] MorphologicalThinning(byte[,] image)
    {
        int height = image.GetLength(0);
        int width = image.GetLength(1);

        // Convert to binary (0 or 1)
        byte[,] binary = new byte[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                binary[y, x] = (byte)(image[y, x] > 127 ? 1 : 0);
            }
        }

        // Apply Zhang-Suen thinning
        bool hasChanged = true;
        while (hasChanged)
        {
            hasChanged = false;

            // Sub-iteration 1
            var toDelete1 = FindPixelsToDelete(binary, 1);
            if (toDelete1.Count > 0)
            {
                foreach (var (y, x) in toDelete1)
                {
                    binary[y, x] = 0;
                }
                hasChanged = true;
            }

            // Sub-iteration 2
            var toDelete2 = FindPixelsToDelete(binary, 2);
            if (toDelete2.Count > 0)
            {
                foreach (var (y, x) in toDelete2)
                {
                    binary[y, x] = 0;
                }
                hasChanged = true;
            }
        }

        // Convert back to 0-255 range
        byte[,] result = new byte[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                result[y, x] = (byte)(binary[y, x] * 255);
            }
        }

        return result;
    }

    /// <summary>
    /// Finds pixels to delete in the current sub-iteration
    /// </summary>
    private static List<(int y, int x)> FindPixelsToDelete(byte[,] image, int iteration)
    {
        int height = image.GetLength(0);
        int width = image.GetLength(1);
        var toDelete = new List<(int, int)>();

        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                // Only consider foreground pixels
                if (image[y, x] != 1)
                    continue;

                // Get 8-connected neighbors (P2-P9 in clockwise order starting from top)
                byte p2 = image[y - 1, x];      // N
                byte p3 = image[y - 1, x + 1];  // NE
                byte p4 = image[y, x + 1];      // E
                byte p5 = image[y + 1, x + 1];  // SE
                byte p6 = image[y + 1, x];      // S
                byte p7 = image[y + 1, x - 1];  // SW
                byte p8 = image[y, x - 1];      // W
                byte p9 = image[y - 1, x - 1];  // NW

                // Condition 1: 2 <= B(P1) <= 6 (number of non-zero neighbors)
                int b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (b < 2 || b > 6)
                    continue;

                // Condition 2: A(P1) = 1 (number of 0->1 transitions in clockwise order)
                int a = CountTransitions(p2, p3, p4, p5, p6, p7, p8, p9);
                if (a != 1)
                    continue;

                // Condition 3 & 4: Different for each sub-iteration
                bool condition3, condition4;

                if (iteration == 1)
                {
                    // Condition 3: P2 * P4 * P6 = 0
                    condition3 = (p2 * p4 * p6) == 0;
                    // Condition 4: P4 * P6 * P8 = 0
                    condition4 = (p4 * p6 * p8) == 0;
                }
                else // iteration == 2
                {
                    // Condition 3: P2 * P4 * P8 = 0
                    condition3 = (p2 * p4 * p8) == 0;
                    // Condition 4: P2 * P6 * P8 = 0
                    condition4 = (p2 * p6 * p8) == 0;
                }

                if (condition3 && condition4)
                {
                    toDelete.Add((y, x));
                }
            }
        }

        return toDelete;
    }

    /// <summary>
    /// Counts 0->1 transitions in clockwise order around the pixel
    /// </summary>
    private static int CountTransitions(byte p2, byte p3, byte p4, byte p5, byte p6, byte p7, byte p8, byte p9)
    {
        int count = 0;

        // Count transitions in clockwise order
        if (p2 == 0 && p3 == 1) count++;
        if (p3 == 0 && p4 == 1) count++;
        if (p4 == 0 && p5 == 1) count++;
        if (p5 == 0 && p6 == 1) count++;
        if (p6 == 0 && p7 == 1) count++;
        if (p7 == 0 && p8 == 1) count++;
        if (p8 == 0 && p9 == 1) count++;
        if (p9 == 0 && p2 == 1) count++;

        return count;
    }
}
