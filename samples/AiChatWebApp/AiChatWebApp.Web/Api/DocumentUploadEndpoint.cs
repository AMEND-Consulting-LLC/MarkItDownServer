using System.Text.RegularExpressions;
using AiChatWebApp.Web.Services;
using AiChatWebApp.Web.Services.Ingestion;

namespace AiChatWebApp.Web.Api;

public static partial class DocumentUploadEndpoint
{
    // Regex to validate file extensions (alphanumeric only)
    [GeneratedRegex(@"^\.[a-zA-Z0-9]+$")]
    private static partial Regex ValidExtensionRegex();

    // Regex to sanitize strings for logging (remove control characters and newlines)
    [GeneratedRegex(@"[\r\n\t\x00-\x1F\x7F]")]
    private static partial Regex LogSanitizeRegex();

    /// <summary>
    /// Sanitizes a string for safe logging by removing control characters and newlines.
    /// </summary>
    private static string SanitizeForLog(string? input)
    {
        if (string.IsNullOrEmpty(input))
            return string.Empty;
        return LogSanitizeRegex().Replace(input, "_");
    }

    /// <summary>
    /// Extracts and validates the file extension, preventing path traversal.
    /// </summary>
    private static string? GetSafeExtension(string? fileName)
    {
        if (string.IsNullOrEmpty(fileName))
            return null;

        // Get only the filename without any path components to prevent path traversal
        var safeFileName = Path.GetFileName(fileName);
        var extension = Path.GetExtension(safeFileName).ToLowerInvariant();

        // Validate extension contains only safe characters
        if (!ValidExtensionRegex().IsMatch(extension))
            return null;

        return extension;
    }

    public static void MapDocumentUploadEndpoint(this IEndpointRouteBuilder app)
    {
        app.MapPost("/api/upload", async (
            IFormFile file,
            IWebHostEnvironment env,
            MarkItDownService markItDownService,
            DataIngestor dataIngestor,
            ILoggerFactory loggerFactory) =>
        {
            var logger = loggerFactory.CreateLogger("DocumentUpload");
            try
            {
                // Validate file
                if (file == null || file.Length == 0)
                {
                    return Results.BadRequest(new { error = "No file provided" });
                }

                // Check file size (50MB limit)
                const long maxFileSize = 50 * 1024 * 1024;
                if (file.Length > maxFileSize)
                {
                    return Results.BadRequest(new { error = "File size exceeds 50MB limit" });
                }

                // Check file extension using safe extraction
                var allowedExtensions = new[] {
                    // Document formats
                    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".txt", ".md", ".html",
                    // Image formats
                    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg",
                    // Audio formats
                    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"
                };
                var extension = GetSafeExtension(file.FileName);
                if (extension == null || !allowedExtensions.Contains(extension))
                {
                    return Results.BadRequest(new { error = $"File type is not supported. Allowed types: {string.Join(", ", allowedExtensions)}" });
                }

                // Create uploads directory with path validation
                if (string.IsNullOrEmpty(env.WebRootPath))
                {
                    return Results.Problem("WebRootPath is not configured", statusCode: 500);
                }
                var uploadsDir = Path.GetFullPath(Path.Combine(env.WebRootPath, "uploads"));

                // Ensure uploads directory is within WebRootPath (prevent path traversal)
                if (!uploadsDir.StartsWith(Path.GetFullPath(env.WebRootPath), StringComparison.OrdinalIgnoreCase))
                {
                    return Results.Problem("Invalid upload directory configuration", statusCode: 500);
                }
                Directory.CreateDirectory(uploadsDir);

                // Save the uploaded file
                var fileName = $"{Guid.NewGuid()}{extension}";
                var filePath = Path.Combine(uploadsDir, fileName);

                using (var stream = new FileStream(filePath, FileMode.Create))
                {
                    await file.CopyToAsync(stream);
                }

                logger.LogInformation("File uploaded: {FileName} ({Length} bytes)", SanitizeForLog(file.FileName), file.Length);

                // Ingest the uploaded file using MarkItDown
                var uploadSource = new UploadedFileSource(
                    uploadsDir,
                    markItDownService,
                    loggerFactory.CreateLogger<UploadedFileSource>());

                await dataIngestor.IngestDataAsync(uploadSource);

                return Results.Ok(new
                {
                    fileName = file.FileName,
                    savedAs = fileName,
                    size = file.Length,
                    message = "File uploaded and processed successfully"
                });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Error uploading file");
                return Results.Problem(
                    title: "Upload failed",
                    detail: ex.Message,
                    statusCode: 500);
            }
        })
        .DisableAntiforgery() // For API endpoint
        .WithName("UploadDocument");
    }
}
