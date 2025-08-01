"use client";

import { useState } from "react";

const ResumeOptimizer = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === "application/pdf") {
      setSelectedFile(file);
      setError(null);
    } else {
      setError("Please select a PDF file");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedFile || !jobDescription.trim()) {
      setError("Please select a PDF file and enter a job description");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("resume_file", selectedFile);
      formData.append("job_description", jobDescription);

      const response = await fetch("/api/v1/resume/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = "Failed to start processing";
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch {
          // If response is not JSON (e.g., HTML error page), use status text
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const job = await response.json();
      setJobId(job.job_id);

      // Poll for results
      pollForResults(job.job_id);
    } catch (err) {
      let errorMessage = "An error occurred";
      if (err instanceof Error) {
        if (err.message.includes("Failed to fetch") || err.message.includes("ECONNREFUSED")) {
          errorMessage = "Backend server is not running. Please start the backend server on port 8000.";
        } else {
          errorMessage = err.message;
        }
      }
      setError(errorMessage);
      setIsProcessing(false);
    }
  };

  const pollForResults = async (jobId: string) => {
    const maxAttempts = 60; // 5 minutes
    let attempts = 0;

    const poll = async () => {
      try {
        const response = await fetch(`/api/v1/resume/status/${jobId}`);

        if (!response.ok) {
          throw new Error(`Status check failed: ${response.status} ${response.statusText}`);
        }

        const status = await response.json();

        if (status.status === "completed") {
          const resultResponse = await fetch(`/api/v1/resume/result/${jobId}`);

          if (!resultResponse.ok) {
            throw new Error(`Result fetch failed: ${resultResponse.status} ${resultResponse.statusText}`);
          }

          const resultData = await resultResponse.json();
          setResult(resultData);
          setIsProcessing(false);
          return;
        }

        if (status.status === "failed") {
          setError(status.error_message || "Processing failed");
          setIsProcessing(false);
          return;
        }

        if (attempts < maxAttempts) {
          attempts++;
          setTimeout(poll, 3000);
        } else {
          setError("Processing timed out");
          setIsProcessing(false);
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Failed to check status";
        setError(`Backend server unavailable: ${errorMessage}`);
        setIsProcessing(false);
      }
    };

    poll();
  };

  const reset = () => {
    setSelectedFile(null);
    setJobDescription("");
    setResult(null);
    setError(null);
    setJobId(null);
    setIsProcessing(false);
  };

  return (
    <section className="pb-16 pt-24 md:pb-20 md:pt-28 lg:pb-24 lg:pt-32">
      <div className="container">
        <div className="mx-auto max-w-4xl">
          <div className="rounded-xs bg-white px-8 py-11 shadow-three dark:bg-gray-dark sm:p-[55px]">
            <h1 className="mb-3 text-2xl font-bold text-black dark:text-white sm:text-3xl">
              Resume Optimizer
            </h1>
            <p className="mb-8 text-base text-body-color">
              Upload your resume and job description to get an ATS-optimized version with preserved layout.
            </p>

            {error && (
              <div className="mb-6 rounded-xs bg-red-100 border border-red-400 text-red-700 px-4 py-3">
                {error}
              </div>
            )}

            {!isProcessing && !result && (
              <form onSubmit={handleSubmit}>
                <div className="mb-6">
                  <label className="mb-3 block text-sm font-medium text-dark dark:text-white">
                    Upload Resume (PDF)
                  </label>
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    className="w-full rounded-xs border border-stroke bg-[#f8f8f8] px-6 py-3 text-base text-body-color outline-none focus:border-primary dark:border-transparent dark:bg-[#2C303B] dark:text-body-color-dark"
                  />
                  {selectedFile && (
                    <p className="mt-2 text-sm text-body-color">
                      Selected: {selectedFile.name}
                    </p>
                  )}
                </div>

                <div className="mb-6">
                  <label className="mb-3 block text-sm font-medium text-dark dark:text-white">
                    Job Description
                  </label>
                  <textarea
                    value={jobDescription}
                    onChange={(e) => setJobDescription(e.target.value)}
                    rows={8}
                    placeholder="Paste the job description here..."
                    className="w-full resize-none rounded-xs border border-stroke bg-[#f8f8f8] px-6 py-3 text-base text-body-color outline-none focus:border-primary dark:border-transparent dark:bg-[#2C303B] dark:text-body-color-dark"
                  />
                </div>

                <button
                  type="submit"
                  disabled={!selectedFile || !jobDescription.trim()}
                  className="rounded-xs bg-primary px-9 py-4 text-base font-medium text-white shadow-submit duration-300 hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Optimize Resume
                </button>
              </form>
            )}

            {isProcessing && (
              <div className="text-center">
                <div className="mb-4">
                  <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-primary border-r-transparent"></div>
                </div>
                <h3 className="mb-2 text-xl font-semibold text-black dark:text-white">
                  Processing Your Resume
                </h3>
                <p className="text-body-color">
                  This may take 1-2 minutes. We&apos;re using AI to optimize your resume while preserving the layout.
                </p>
                {jobId && (
                  <p className="mt-2 text-sm text-body-color">
                    Job ID: {jobId}
                  </p>
                )}
              </div>
            )}

            {result && (
              <div>
                <h3 className="mb-4 text-xl font-semibold text-black dark:text-white">
                  Optimization Complete! 🎉
                </h3>

                {result.resume_analysis && (
                  <div className="mb-6 rounded-xs bg-green-50 border border-green-200 p-4 dark:bg-green-900/20 dark:border-green-500">
                    <p className="text-sm text-green-700 dark:text-green-400">
                      <strong>Layout Method:</strong> {result.resume_analysis.layout_method} |
                      <strong> Blocks Found:</strong> {result.resume_analysis.blocks_found} |
                      <strong> Processing Time:</strong> {result.processing_time_seconds?.toFixed(1)}s
                    </p>
                  </div>
                )}

                <div className="space-y-4">
                  {result.optimized_resume_blob_url && (
                    <a
                      href={result.optimized_resume_blob_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-block rounded-xs bg-primary px-6 py-3 text-white hover:bg-primary/90 mr-4"
                    >
                      📄 Download Optimized Resume
                    </a>
                  )}

                  {result.cover_letter_blob_url && (
                    <a
                      href={result.cover_letter_blob_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-block rounded-xs bg-secondary px-6 py-3 text-white hover:bg-secondary/90 mr-4"
                    >
                      📝 Download Cover Letter
                    </a>
                  )}

                  {result.package_zip_blob_url && (
                    <a
                      href={result.package_zip_blob_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-block rounded-xs bg-green-600 px-6 py-3 text-white hover:bg-green-700 mr-4"
                    >
                      📦 Download Complete Package
                    </a>
                  )}
                </div>

                <button
                  onClick={reset}
                  className="mt-6 rounded-xs border border-stroke px-6 py-3 text-body-color hover:border-primary hover:text-primary"
                >
                  Optimize Another Resume
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ResumeOptimizer;
