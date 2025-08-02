"use client";

import { useState, useRef, useCallback } from "react";

interface ProcessingStep {
  id: string;
  name: string;
  status: 'pending' | 'active' | 'complete' | 'error';
  description: string;
}

const ResumeOptimizer = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [companyName, setCompanyName] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([
    { id: 'upload', name: 'File Upload', status: 'pending', description: 'Uploading resume file' },
    { id: 'extract', name: 'Text Extraction', status: 'pending', description: 'Extracting text from PDF' },
    { id: 'optimize', name: 'AI Optimization', status: 'pending', description: 'Optimizing content with AI' },
    { id: 'generate', name: 'PDF Generation', status: 'pending', description: 'Creating optimized documents' }
  ]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): string | null => {
    if (!file) return "No file selected";
    if (file.type !== "application/pdf") return "Please select a PDF file";
    if (file.size > 10 * 1024 * 1024) return "File size must be less than 10MB";
    if (file.size < 1024) return "File appears to be too small";
    return null;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        setSelectedFile(null);
      } else {
        setSelectedFile(file);
        setError(null);
      }
    }
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        setSelectedFile(null);
      } else {
        setSelectedFile(file);
        setError(null);
      }
    }
  }, []);

  const updateProcessingStep = (stepId: string, status: ProcessingStep['status']) => {
    setProcessingSteps(prev => prev.map(step =>
      step.id === stepId ? { ...step, status } : step
    ));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedFile || !jobDescription.trim()) {
      setError("Please select a PDF file and enter a job description");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);

    // Reset processing steps
    setProcessingSteps(prev => prev.map(step => ({ ...step, status: 'pending' })));

    try {
      // Step 1: File Upload
      updateProcessingStep('upload', 'active');

      const formData = new FormData();
      formData.append("resume_file", selectedFile);
      formData.append("job_description", jobDescription);

      // Add optional fields if provided
      if (jobTitle.trim()) {
        formData.append("job_title", jobTitle);
      }
      if (companyName.trim()) {
        formData.append("company_name", companyName);
      }

      updateProcessingStep('upload', 'complete');
      updateProcessingStep('extract', 'active');

      // Use the smart models API on port 8000
      const response = await fetch("http://localhost:8000/optimize", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = "Failed to process resume";
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch {
          // If response is not JSON (e.g., HTML error page), use status text
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      updateProcessingStep('extract', 'complete');
      updateProcessingStep('optimize', 'active');

      const result = await response.json();

      updateProcessingStep('optimize', 'complete');
      updateProcessingStep('generate', 'active');

      // Simulate brief delay for PDF generation step
      await new Promise(resolve => setTimeout(resolve, 500));

      updateProcessingStep('generate', 'complete');

      // Set result and complete processing
      setResult(result);
      setIsProcessing(false);
    } catch (err) {
      // Mark current step as error
      const currentStep = processingSteps.find(step => step.status === 'active');
      if (currentStep) {
        updateProcessingStep(currentStep.id, 'error');
      }

      let errorMessage = "An error occurred";
      if (err instanceof Error) {
        if (err.message.includes("Failed to fetch") || err.message.includes("ECONNREFUSED")) {
          errorMessage = "API server is not running. Please start the API server on port 8000.";
        } else {
          errorMessage = err.message;
        }
      }
      setError(errorMessage);
      setIsProcessing(false);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setJobDescription("");
    setJobTitle("");
    setCompanyName("");
    setResult(null);
    setError(null);
    setJobId(null);
    setIsProcessing(false);
    setProcessingSteps(prev => prev.map(step => ({ ...step, status: 'pending' })));
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <section className="pb-16 pt-24 md:pb-20 md:pt-28 lg:pb-24 lg:pt-32">
      <div className="container">
        <div className="mx-auto max-w-6xl">
          <div className="rounded-xs bg-white px-8 py-11 shadow-three dark:bg-gray-dark sm:p-[55px]">
            <div className="text-center mb-8">
              <h1 className="mb-3 text-3xl font-bold text-black dark:text-white sm:text-4xl">
                AI Resume Optimizer
              </h1>
              <p className="text-lg text-body-color max-w-2xl mx-auto">
                Upload your resume and job description to get an ATS-optimized version with advanced keyword optimization and preserved layout.
              </p>
            </div>

            {error && (
              <div className="mb-6 rounded-lg bg-red-50 border border-red-200 text-red-800 px-4 py-3 dark:bg-red-900/20 dark:border-red-500 dark:text-red-300">
                <div className="flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  {error}
                </div>
              </div>
            )}

            {!isProcessing && !result && (
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Enhanced File Upload Area */}
                <div className="mb-6">
                  <label className="mb-3 block text-sm font-medium text-dark dark:text-white">
                    Upload Resume (PDF) *
                  </label>
                  <div
                    className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
                      dragActive
                        ? 'border-primary bg-primary/5'
                        : selectedFile
                        ? 'border-green-400 bg-green-50 dark:bg-green-900/20'
                        : 'border-gray-300 hover:border-primary dark:border-gray-600'
                    }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf"
                      onChange={handleFileChange}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />

                    {selectedFile ? (
                      <div className="space-y-2">
                        <div className="flex items-center justify-center">
                          <svg className="w-8 h-8 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                        <p className="text-sm font-medium text-green-700 dark:text-green-300">
                          {selectedFile.name}
                        </p>
                        <p className="text-xs text-green-600 dark:text-green-400">
                          {formatFileSize(selectedFile.size)}
                        </p>
                        <button
                          type="button"
                          onClick={() => {
                            setSelectedFile(null);
                            if (fileInputRef.current) fileInputRef.current.value = '';
                          }}
                          className="text-xs text-red-600 hover:text-red-800 dark:text-red-400"
                        >
                          Remove file
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <div className="flex items-center justify-center">
                          <svg className="w-12 h-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          <span className="font-medium text-primary cursor-pointer hover:underline">
                            Click to upload
                          </span> or drag and drop your resume
                        </p>
                        <p className="text-xs text-gray-500">
                          PDF files only, max 10MB
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Optional Fields */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <div>
                    <label className="mb-3 block text-sm font-medium text-dark dark:text-white">
                      Job Title (Optional)
                    </label>
                    <input
                      type="text"
                      value={jobTitle}
                      onChange={(e) => setJobTitle(e.target.value)}
                      placeholder="e.g., Senior Software Engineer"
                      className="w-full rounded-lg border border-stroke bg-[#f8f8f8] px-4 py-3 text-base text-body-color outline-none focus:border-primary dark:border-transparent dark:bg-[#2C303B] dark:text-body-color-dark"
                    />
                  </div>
                  <div>
                    <label className="mb-3 block text-sm font-medium text-dark dark:text-white">
                      Company Name (Optional)
                    </label>
                    <input
                      type="text"
                      value={companyName}
                      onChange={(e) => setCompanyName(e.target.value)}
                      placeholder="e.g., Google, Microsoft"
                      className="w-full rounded-lg border border-stroke bg-[#f8f8f8] px-4 py-3 text-base text-body-color outline-none focus:border-primary dark:border-transparent dark:bg-[#2C303B] dark:text-body-color-dark"
                    />
                  </div>
                </div>

                {/* Job Description */}
                <div className="mb-6">
                  <label className="mb-3 block text-sm font-medium text-dark dark:text-white">
                    Job Description *
                  </label>
                  <textarea
                    value={jobDescription}
                    onChange={(e) => setJobDescription(e.target.value)}
                    placeholder="Paste the complete job description here. Include requirements, responsibilities, and preferred qualifications for better optimization..."
                    rows={10}
                    className="w-full rounded-lg border border-stroke bg-[#f8f8f8] px-4 py-3 text-base text-body-color outline-none focus:border-primary dark:border-transparent dark:bg-[#2C303B] dark:text-body-color-dark resize-none"
                  />
                  <div className="mt-2 flex justify-between items-center">
                    <p className="text-xs text-gray-500">
                      {jobDescription.length} characters
                    </p>
                    {jobDescription.length > 0 && (
                      <button
                        type="button"
                        onClick={() => setJobDescription("")}
                        className="text-xs text-gray-500 hover:text-red-600"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                </div>

                {/* Submit Button */}
                <div className="text-center">
                  <button
                    type="submit"
                    disabled={!selectedFile || !jobDescription.trim()}
                    className="inline-flex items-center px-8 py-4 text-base font-medium text-white bg-primary rounded-lg shadow-lg hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
                  >
                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Optimize Resume with AI
                  </button>
                  <p className="mt-2 text-sm text-gray-500">
                    Processing typically takes 30-60 seconds
                  </p>
                </div>
              </form>
            )}

            {isProcessing && (
              <div className="max-w-2xl mx-auto">
                <div className="text-center mb-8">
                  <div className="mb-4">
                    <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-solid border-primary border-r-transparent"></div>
                  </div>
                  <h3 className="mb-2 text-2xl font-semibold text-black dark:text-white">
                    Optimizing Your Resume
                  </h3>
                  <p className="text-body-color">
                    Our AI is analyzing your resume and creating an optimized version. This typically takes 30-60 seconds.
                  </p>
                </div>

                {/* Processing Steps */}
                <div className="space-y-4">
                  {processingSteps.map((step, index) => (
                    <div key={step.id} className="flex items-center space-x-4">
                      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                        step.status === 'complete'
                          ? 'bg-green-500 text-white'
                          : step.status === 'active'
                          ? 'bg-primary text-white'
                          : step.status === 'error'
                          ? 'bg-red-500 text-white'
                          : 'bg-gray-200 text-gray-500 dark:bg-gray-700'
                      }`}>
                        {step.status === 'complete' ? (
                          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        ) : step.status === 'active' ? (
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        ) : step.status === 'error' ? (
                          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                          </svg>
                        ) : (
                          <span className="text-sm font-medium">{index + 1}</span>
                        )}
                      </div>
                      <div className="flex-1">
                        <p className={`text-sm font-medium ${
                          step.status === 'complete'
                            ? 'text-green-700 dark:text-green-300'
                            : step.status === 'active'
                            ? 'text-primary'
                            : step.status === 'error'
                            ? 'text-red-700 dark:text-red-300'
                            : 'text-gray-500'
                        }`}>
                          {step.name}
                        </p>
                        <p className="text-xs text-gray-500">
                          {step.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>

                {jobId && (
                  <div className="mt-6 text-center">
                    <p className="text-xs text-gray-500">
                      Session ID: {jobId}
                    </p>
                  </div>
                )}
              </div>
            )}

            {result && (
              <div>
                <h3 className="mb-4 text-xl font-semibold text-black dark:text-white">
                  Optimization Complete! 🎉
                </h3>

                {/* AI Processing Status */}
                {result.ai_processing && (
                  <div className="mb-6 rounded-xs bg-blue-50 border border-blue-200 p-4 dark:bg-blue-900/20 dark:border-blue-500">
                    <h4 className="text-sm font-medium text-blue-800 dark:text-blue-300 mb-2">
                      AI Processing Status:
                    </h4>
                    <div className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                      <p>
                        Resume: {result.ai_processing.used_fallback_resume ?
                          "⚠️ Used fallback (AI unavailable)" :
                          "✅ AI optimized"}
                      </p>
                      <p>
                        Cover Letter: {result.ai_processing.used_fallback_cover_letter ?
                          "⚠️ Used fallback (AI unavailable)" :
                          "✅ AI generated"}
                      </p>
                      <p>Model: {result.ai_processing.model_used}</p>
                    </div>
                  </div>
                )}

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
                  {result.downloads?.optimized_resume_pdf && (
                    <a
                      href={`http://localhost:8000${result.downloads.optimized_resume_pdf}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-block rounded-xs bg-primary px-6 py-3 text-white hover:bg-primary/90 mr-4"
                    >
                      📄 Download Optimized Resume PDF
                    </a>
                  )}

                  {result.downloads?.cover_letter_pdf && (
                    <a
                      href={`http://localhost:8000${result.downloads.cover_letter_pdf}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-block rounded-xs bg-secondary px-6 py-3 text-white hover:bg-secondary/90 mr-4"
                    >
                      📝 Download Cover Letter PDF
                    </a>
                  )}

                  {result.downloads?.optimized_resume_markdown && (
                    <a
                      href={`http://localhost:8000${result.downloads.optimized_resume_markdown}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-block rounded-xs bg-green-600 px-6 py-3 text-white hover:bg-green-700 mr-4"
                    >
                      📄 Download Resume Markdown
                    </a>
                  )}

                  {result.downloads?.cover_letter_markdown && (
                    <a
                      href={`http://localhost:8000${result.downloads.cover_letter_markdown}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-block rounded-xs bg-blue-600 px-6 py-3 text-white hover:bg-blue-700 mr-4"
                    >
                      📝 Download Cover Letter Markdown
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
