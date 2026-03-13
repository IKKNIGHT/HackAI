import { useState, useRef, useEffect, useCallback } from "react";
import {
  Check,
  AlertCircle,
  Loader2,
  Upload,
  Info,
  ImageIcon,
  Download,
  FlaskConical,
  ChevronDown,
  Camera,
  X,
} from "lucide-react";
import { ModelState } from "../App";
import { predictCSV, predictImage, startGitHubAuth } from "../services/api";

type TrainingMode = "tabular" | "image";

interface ControlsPanelProps {
  trees: number;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  trainingMode: TrainingMode;
  modelState: ModelState;
}

interface PredictionResult {
  prediction: string;
  confidence?: number;
  probabilities?: number[];
}

export function ControlsPanel({
  trees,
  maxDepth,
  minSamplesSplit,
  minSamplesLeaf,
  trainingMode,
  modelState,
}: ControlsPanelProps) {
  const {
    isTrained,
    isTraining,
    metrics,
    mode,
    dataType,
    features,
    classes,
    error,
    featureImportance,
    modelExplanation: cachedExplanation,
  } = modelState;

  // Prediction state
  const [predictionResult, setPredictionResult] =
    useState<PredictionResult | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [isDemoing, setIsDemoing] = useState(false);
  const [showDemoPopup, setShowDemoPopup] = useState(false);
  // Only allow collapse in CSV mode
  const [testPredictionExpanded, setTestPredictionExpanded] = useState(true);
  const [demoPrompt, setDemoPrompt] = useState("");

  // CSV form state - dynamic based on features
  const [formData, setFormData] = useState<Record<string, string>>({});

  // Image state
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  // Webcam state
  const [showWebcam, setShowWebcam] = useState(false);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Reset form when features change
  useEffect(() => {
    if (features.length > 0) {
      const initialData: Record<string, string> = {};
      features.forEach((f) => {
        initialData[f] = "";
      });
      setFormData(initialData);
    }
  }, [features]);

  // Reset prediction when mode changes
  useEffect(() => {
    setPredictionResult(null);
    setPredictionError(null);
  }, [trainingMode]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImageFile(file);
    setPredictionResult(null);
    setPredictionError(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  // Webcam functions
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 640, height: 480 },
      });
      setWebcamStream(stream);
      setShowWebcam(true);
    } catch (err) {
      setPredictionError("Could not access webcam. Please check permissions.");
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      setWebcamStream(null);
    }
    setShowWebcam(false);
  }, [webcamStream]);

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;

        const file = new File([blob], "webcam-capture.jpg", {
          type: "image/jpeg",
        });
        setImageFile(file);
        setImagePreview(canvas.toDataURL("image/jpeg"));
        setPredictionResult(null);
        setPredictionError(null);
        stopWebcam();
      },
      "image/jpeg",
      0.9,
    );
  }, [stopWebcam]);

  // Cleanup webcam on unmount
  useEffect(() => {
    return () => {
      if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [webcamStream]);

  // Connect stream to video element when both are available
  useEffect(() => {
    if (videoRef.current && webcamStream) {
      videoRef.current.srcObject = webcamStream;
    }
  }, [webcamStream, showWebcam]);

  const handlePredictCSV = async () => {
    if (!isTrained || dataType !== "csv") {
      setPredictionError("Please train a CSV model first");
      return;
    }

    // Validate form data
    const emptyFields = features.filter((f) => !formData[f]?.trim());
    if (emptyFields.length > 0) {
      setPredictionError(`Please fill in: ${emptyFields.join(", ")}`);
      return;
    }

    setIsPredicting(true);
    setPredictionError(null);

    try {
      const result = await predictCSV(formData);
      setPredictionResult({
        prediction: result.prediction,
        probabilities: result.probabilities,
        confidence: result.probabilities
          ? Math.max(...result.probabilities) * 100
          : undefined,
      });
      setTestPredictionExpanded(false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setPredictionError(
        err instanceof Error ? err.message : "Prediction failed",
      );
    } finally {
      setIsPredicting(false);
    }
  };

  const handlePredictImage = async () => {
    if (!isTrained || dataType !== "image") {
      setPredictionError("Please train an image model first");
      return;
    }

    if (!imageFile) {
      setPredictionError("Please upload an image");
      return;
    }

    setIsPredicting(true);
    setPredictionError(null);

    try {
      const result = await predictImage(imageFile);
      setPredictionResult({
        prediction: result.prediction,
      });
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setPredictionError(
        err instanceof Error ? err.message : "Prediction failed",
      );
    } finally {
      setIsPredicting(false);
    }
  };

  let isCSVMode = trainingMode === "tabular";
  const canPredict =
    isTrained && (isCSVMode ? dataType === "csv" : dataType === "image");

  const handleExportModel = async () => {
    setIsExporting(true);
    setPredictionError(null);
    try {
      const response = await fetch("http://localhost:5000/export_model");
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Export failed");
      }
      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition") ?? "";
      const filename =
        disposition.match(/filename=([^\s;]+)/)?.[1] ??
        `trained_model_${dataType}.pkl`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setPredictionError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setIsExporting(false);
    }
  };

  const handleDemo = async () => {
    setShowDemoPopup(true);
  };

  const handleDemoConfirm = () => {
    if (!demoPrompt.trim()) return;
    setShowDemoPopup(false);
    startGitHubAuth(demoPrompt);
    setDemoPrompt("");
  };

  // Process feature importance data - take top 6 and normalize
  const processedFeatures =
    featureImportance.length > 0
      ? featureImportance.slice(0, 6).map(([name, value]) => ({
          name: name.length > 15 ? name.substring(0, 12) + "..." : name,
          value: value,
        }))
      : [];

  // Find max for normalization
  const maxImportance =
    processedFeatures.length > 0
      ? Math.max(...processedFeatures.map((f) => f.value))
      : 1;

  // Calculate accuracy for display
  const accuracy = metrics?.accuracy ?? metrics?.r2_score ?? 0;
  const accuracyPercent = Math.round(accuracy * 100);
  const isRegression = mode === "Regression";

  // In image mode, always expanded
  isCSVMode = trainingMode === "tabular";
  useEffect(() => {
    if (!isCSVMode) setTestPredictionExpanded(true);
  }, [isCSVMode]);

  return (
    <div className="flex flex-col gap-4 h-full overflow-y-auto overflow-x-hidden px-2 pb-4">
      {/* Demo Confirmation Popup */}
      {showDemoPopup && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="relative w-[450px] p-5 rounded-xl bg-[#1a1a2e] border border-white/20 shadow-2xl">
            {/* Close X button */}
            <button
              onClick={() => setShowDemoPopup(false)}
              className="absolute top-3 right-3 p-1 rounded-lg hover:bg-white/10 transition-colors"
            >
              <svg
                className="w-5 h-5 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>

            {/* Header */}
            <h3 className="text-lg font-bold text-white mb-2">
              Generate & Deploy
            </h3>
            <p className="text-gray-400 text-sm mb-4">
              Describe what you want to build with your trained model. We'll
              generate the code and push it to a new GitHub repository.
            </p>

            {/* Model Explanation from Gemini */}
            <div className="mb-4 p-3 rounded-lg bg-[#00F0FF]/10 border border-[#00F0FF]/30">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-[#00F0FF] text-sm font-semibold">
                  AI Model Summary
                </span>
              </div>
              {cachedExplanation ? (
                <p className="text-gray-300 text-sm">{cachedExplanation}</p>
              ) : (
                <p className="text-gray-400 text-sm italic">
                  Explanation not available
                </p>
              )}
            </div>

            {/* Prompt Input */}
            <label className="block text-sm font-medium text-gray-300 mb-2">
              What would you like to build?
            </label>
            <textarea
              value={demoPrompt}
              onChange={(e) => setDemoPrompt(e.target.value)}
              placeholder="Describe the application you want to create..."
              className="w-full h-28 px-3 py-2 rounded-lg bg-white/5 border border-white/20 text-white text-sm placeholder-gray-500 resize-none focus:outline-none focus:ring-2 focus:ring-[#00F0FF] focus:border-transparent"
            />
            <p className="text-xs text-gray-500 mt-2 italic">
              Example: "Create a Flask web app with a clean UI that lets users
              upload data and get predictions."
            </p>

            {/* Buttons */}
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={() => setShowDemoPopup(false)}
                className="px-4 py-2 rounded-lg font-medium text-sm text-gray-300 bg-white/5 hover:bg-white/10 transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleDemoConfirm}
                disabled={!demoPrompt.trim()}
                className={`px-4 py-2 rounded-lg font-bold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed ${
                  isCSVMode
                    ? "bg-[#39FF14] hover:bg-[#39FF14]/90 text-black shadow-[0_0_15px_rgba(57,255,20,0.3)]"
                    : "bg-[#00F0FF] hover:bg-[#00F0FF]/90 text-black shadow-[0_0_15px_rgba(0,240,255,0.3)]"
                }`}
              >
                Generate & Push
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Test Prediction */}
      <div className="p-4 rounded-xl z-30 bg-white/5 backdrop-blur-sm border border-white/10 overflow-visible">
        {isCSVMode ? (
          <button
            onClick={() => setTestPredictionExpanded(!testPredictionExpanded)}
            className="w-full flex items-center justify-between text-xs font-semibold text-gray-400 mb-3 uppercase"
          >
            <span>Test Prediction</span>
            <ChevronDown
              className={`w-4 h-4 transition-transform ${testPredictionExpanded ? "rotate-180" : ""}`}
            />
          </button>
        ) : (
          <div className="w-full flex items-center justify-between text-xs font-semibold text-gray-400 mb-3 uppercase">
            <span>Test Prediction</span>
          </div>
        )}

        {testPredictionExpanded && !isTrained && (
          <div className="flex items-center gap-2 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30 mb-3">
            <AlertCircle className="w-4 h-4 text-yellow-500" />
            <span className="text-xs text-yellow-500">Train a model first</span>
          </div>
        )}

        {testPredictionExpanded && isTrained && !canPredict && (
          <div className="flex items-center gap-2 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30 mb-3">
            <AlertCircle className="w-4 h-4 text-yellow-500" />
            <span className="text-xs text-yellow-500">
              Model is for {dataType === "csv" ? "tabular" : "image"} data
            </span>
          </div>
        )}

        {isCSVMode ? (
          <>
            {/* Dynamic Form Fields - Collapsible */}
            {testPredictionExpanded &&
              (features.length > 0 ? (
                <div className="space-y-3 mb-4">
                  {features.map((feature) => (
                    <div key={feature} className="group">
                      <label
                        className="text-xs font-medium text-gray-400 mb-1.5 block truncate uppercase tracking-wide"
                        title={feature}
                      >
                        {feature}
                      </label>
                      <input
                        type="text"
                        value={formData[feature] || ""}
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            [feature]: e.target.value,
                          })
                        }
                        disabled={!canPredict || isPredicting}
                        placeholder="Enter value..."
                        className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:border-[#39FF14]/50 focus:bg-white/10 focus:shadow-[0_0_15px_rgba(57,255,20,0.15)] disabled:opacity-50 disabled:cursor-not-allowed text-sm transition-all duration-200 hover:border-white/20 hover:bg-white/[0.07]"
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 text-xs mb-3">
                  Train a model to see input fields
                </p>
              ))}

            {/* Predict Button - Always visible */}
            <button
              onClick={handlePredictCSV}
              disabled={!canPredict || isPredicting}
              className="w-full px-4 py-2 rounded-lg bg-[#39FF14] hover:bg-[#39FF14]/90 text-black font-bold text-sm shadow-[0_0_20px_rgba(57,255,20,0.4)] hover:shadow-[0_0_30px_rgba(57,255,20,0.6)] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isPredicting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  PREDICTING...
                </>
              ) : (
                "PREDICT"
              )}
            </button>
          </>
        ) : (
          <>
            {/* Webcam Modal */}
            {showWebcam && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
                <div className="relative w-[500px] p-4 rounded-xl bg-[#1a1a2e] border border-white/20 shadow-2xl">
                  <button
                    onClick={stopWebcam}
                    className="absolute top-3 right-3 p-1 rounded-lg hover:bg-white/10 transition-colors z-10"
                  >
                    <X className="w-5 h-5 text-gray-400" />
                  </button>

                  <h3 className="text-lg font-bold text-white mb-3">
                    Take a Photo
                  </h3>

                  <div className="relative rounded-lg overflow-hidden bg-black mb-3">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-auto"
                    />
                  </div>

                  <canvas ref={canvasRef} className="hidden" />

                  <button
                    onClick={capturePhoto}
                    className="w-full px-4 py-3 rounded-lg bg-[#00F0FF] hover:bg-[#00F0FF]/90 text-black font-bold text-sm shadow-[0_0_20px_rgba(0,240,255,0.4)] transition-all flex items-center justify-center gap-2"
                  >
                    <Camera className="w-5 h-5" />
                    Capture Photo
                  </button>
                </div>
              </div>
            )}

            {/* Image Upload - Collapsible */}
            {testPredictionExpanded && (
              <div className="mb-3">
                <input
                  type="file"
                  ref={imageInputRef}
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
                <div className="flex gap-2">
                  <button
                    onClick={() => imageInputRef.current?.click()}
                    disabled={!canPredict || isPredicting}
                    className="flex-1 px-3 py-2 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20 text-white font-medium flex items-center justify-center gap-2 transition-all disabled:opacity-50 text-sm"
                  >
                    <Upload className="w-4 h-4" />
                    Choose Image
                  </button>
                  <button
                    onClick={startWebcam}
                    disabled={!canPredict || isPredicting}
                    className="px-3 py-2 rounded-lg bg-[#00F0FF]/20 hover:bg-[#00F0FF]/30 border border-[#00F0FF]/40 text-[#00F0FF] font-medium flex items-center justify-center gap-2 transition-all disabled:opacity-50 text-sm"
                  >
                    <Camera className="w-4 h-4" />
                    Webcam
                  </button>
                </div>

                {/* Image Preview */}
                {imagePreview && (
                  <div className="relative mt-2 w-full h-24 rounded-lg bg-white/5 border border-white/10 overflow-hidden">
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="w-full h-full object-contain"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setImagePreview(null);
                        setImageFile(null);
                      }}
                      className="absolute top-1 right-1 p-1 rounded-full bg-black/40 hover:bg-black/70 text-white z-10"
                      aria-label="Remove preview"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                )}

                {imageFile && (
                  <p className="text-xs text-gray-400 mt-1 truncate">
                    {imageFile.name}
                  </p>
                )}
              </div>
            )}

            {/* Available Classes */}
            {testPredictionExpanded && classes.length > 0 && (
              <p className="text-xs text-gray-400 mb-3">
                Classes: {classes.join(", ")}
              </p>
            )}

            {/* Predict Button - Always visible */}
            <button
              onClick={handlePredictImage}
              disabled={!canPredict || isPredicting || !imageFile}
              className="w-full px-4 py-2 rounded-lg bg-[#00F0FF] hover:bg-[#00F0FF]/90 text-black font-bold text-sm shadow-[0_0_20px_rgba(0,240,255,0.4)] hover:shadow-[0_0_30px_rgba(0,240,255,0.6)] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isPredicting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  PREDICTING...
                </>
              ) : (
                "PREDICT"
              )}
            </button>
          </>
        )}

        {/* Export & Demo Buttons - Single unified section */}
        {isTrained && (
          <div className="flex gap-2 mt-2">
            <button
              onClick={handleExportModel}
              disabled={isExporting}
              className={`flex-1 px-4 py-2 rounded-lg font-bold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 border ${
                isCSVMode
                  ? "border-[#39FF14]/40 text-[#39FF14] bg-[#39FF14]/10 hover:bg-[#39FF14]/20"
                  : "border-[#00F0FF]/40 text-[#00F0FF] bg-[#00F0FF]/10 hover:bg-[#00F0FF]/20"
              }`}
            >
              {isExporting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  EXPORTING...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  EXPORT
                </>
              )}
            </button>
            <button
              onClick={handleDemo}
              disabled={isDemoing}
              className="flex-1 px-4 py-2 rounded-lg font-bold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 border border-white/20 text-gray-300 bg-white/5 hover:bg-white/10"
            >
              {isDemoing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  LOADING...
                </>
              ) : (
                <>
                  <FlaskConical className="w-4 h-4" />
                  DEMO
                </>
              )}
            </button>
          </div>
        )}

        {/* Error Display */}
        {predictionError && (
          <div className="mt-3 p-2 rounded-lg bg-red-500/10 border border-red-500/30 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
            <span className="text-xs text-red-500">{predictionError}</span>
          </div>
        )}

        {/* Prediction Results */}
        {predictionResult && (
          <div
            className={`relative mt-3 p-3 rounded-lg border ${
              isCSVMode
                ? "bg-gradient-to-br from-[#39FF14]/10 to-[#39FF14]/5 border-[#39FF14]/30"
                : "bg-gradient-to-br from-[#00F0FF]/10 to-[#00F0FF]/5 border-[#00F0FF]/30"
            }`}
          >
            <button
              type="button"
              onClick={() => setPredictionResult(null)}
              className="absolute top-2 right-2 p-1 rounded-full bg-black/40 hover:bg-black/70 text-white z-10"
              aria-label="Hide prediction results"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="flex items-start justify-between mb-2">
              <div>
                <h4 className="text-sm font-semibold text-white">
                  {predictionResult.prediction}
                </h4>
                {predictionResult.confidence !== undefined && (
                  <p className="text-xs text-gray-300">
                    {predictionResult.confidence.toFixed(1)}% confidence
                  </p>
                )}
              </div>
              {predictionResult.probabilities &&
                predictionResult.probabilities.length > 0 && (
                  <button className="p-1 hover:bg-white/10 rounded transition-colors group relative">
                    <Info className="w-4 h-4 text-gray-400" />
                    <div className="absolute right-0 bottom-full mb-1 w-48 p-2 rounded-lg bg-black/90 border border-white/20 text-xs text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                      <div className="font-semibold text-white mb-1">
                        Class Probabilities:
                      </div>
                      <div className="space-y-0.5">
                        {predictionResult.probabilities.map((prob, idx) => (
                          <div key={idx}>
                            Class {idx}: {(prob * 100).toFixed(1)}%
                          </div>
                        ))}
                      </div>
                    </div>
                  </button>
                )}
            </div>

            {/* Confidence Bar */}
            {predictionResult.confidence !== undefined && (
              <div className="relative h-2 rounded-full bg-white/10 overflow-hidden">
                <div
                  className={`h-full transition-all duration-1000 ${
                    isCSVMode
                      ? "bg-gradient-to-r from-[#39FF14] to-[#2acc0f]"
                      : "bg-gradient-to-r from-[#00F0FF] to-[#00c4cc]"
                  }`}
                  style={{ width: `${predictionResult.confidence}%` }}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Performance Card */}
      <div className="p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
        <h3 className="text-xs font-semibold text-gray-400 mb-3 uppercase">
          Performance
        </h3>

        {/* Progress Ring */}
        <div className="flex flex-col items-center mb-4">
          <div className="relative w-32 h-32">
            <svg className="w-32 h-32 transform -rotate-90">
              <circle
                cx="64"
                cy="64"
                r="56"
                stroke="rgba(255,255,255,0.1)"
                strokeWidth="8"
                fill="none"
              />
              {isTrained && (
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="#39FF14"
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${2 * Math.PI * 56}`}
                  strokeDashoffset={`${2 * Math.PI * 56 * (1 - accuracy)}`}
                  className="drop-shadow-[0_0_10px_rgba(57,255,20,0.8)] transition-all duration-1000"
                  strokeLinecap="round"
                />
              )}
            </svg>
            <div className="absolute inset-0 flex items-center justify-center flex-col">
              {isTraining ? (
                <Loader2 className="w-8 h-8 text-[#39FF14] animate-spin" />
              ) : isTrained ? (
                <>
                  <div className="text-3xl font-bold text-white">
                    {dataType === "image" ? "N/A" : isRegression ? accuracy.toFixed(2) : `${accuracyPercent}%`}
                  </div>
                  <div className="text-xs text-gray-400">
                    {dataType === "image" ? "Image Model" : isRegression ? "R² Score" : "Accuracy"}
                  </div>
                </>
              ) : (
                <>
                  <div className="text-2xl font-bold text-gray-500">--</div>
                  <div className="text-xs text-gray-500">Not trained</div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Status */}
        {error ? (
          <div className="flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/50 mb-3">
            <AlertCircle className="w-4 h-4 text-red-500" />
            <span className="text-sm font-medium text-red-500 truncate">
              {error}
            </span>
          </div>
        ) : isTraining ? (
          <div className="flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/50 mb-3">
            <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />
            <span className="text-sm font-medium text-yellow-500">
              Training in progress...
            </span>
          </div>
        ) : isTrained ? (
          <div className="flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-[#39FF14]/10 border border-[#39FF14]/50 mb-3">
            <Check className="w-4 h-4 text-[#39FF14]" />
            <span className="text-sm font-medium text-[#39FF14]">
              Training Complete
            </span>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-white/5 border border-white/20 mb-3">
            <span className="text-sm font-medium text-gray-400">
              Ready to train
            </span>
          </div>
        )}

        {/* Additional Metrics */}
        {isTrained && metrics && (
          <div className="space-y-1">
            {metrics.mse !== undefined && (
              <div className="flex justify-between text-sm">
                <span className="text-gray-300">MSE:</span>
                <span className="text-white font-semibold">
                  {metrics.mse.toFixed(4)}
                </span>
              </div>
            )}
            {metrics.samples !== undefined && (
              <div className="flex justify-between text-sm">
                <span className="text-gray-300">Samples:</span>
                <span className="text-white font-semibold">
                  {metrics.samples}
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Model Info Card */}
      <div className="p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
        <h3 className="text-xs font-semibold text-gray-400 mb-3 uppercase">
          Model Info
        </h3>
        <div className="space-y-2">
          <div className="text-white font-semibold">
            Random Forest {mode || "Classifier"}
          </div>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between text-gray-300">
              <span>Number of trees:</span>
              <span className="text-white font-medium">{trees}</span>
            </div>
            <div className="flex justify-between text-gray-300">
              <span>Max depth:</span>
              <span className="text-white font-medium">{maxDepth}</span>
            </div>
            <div className="flex justify-between text-gray-300">
              <span>{dataType === "image" ? "Classes:" : "Features:"}</span>
              <span className="text-white font-medium">
                {dataType === "image" ? classes.length : features.length || "-"}
              </span>
            </div>
            {dataType === "csv" && mode === "Classification" && (
              <div className="flex justify-between text-gray-300">
                <span>Type:</span>
                <span className="text-white font-medium">{mode}</span>
              </div>
            )}
            {dataType === "image" && classes.length > 0 && (
              <div className="mt-2 pt-2 border-t border-white/10">
                <span className="text-gray-400 text-xs">Classes: </span>
                <span className="text-white text-xs">{classes.join(", ")}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10">
        <h3 className="text-xs font-semibold text-gray-400 mb-3 uppercase">
          {dataType === "image" ? "Image Classification" : "Feature Importance"}
        </h3>
        {dataType === "image" && isTrained ? (
          <p className="text-sm text-gray-300">
            Using MobileNetV2 feature extraction (1280 features)
          </p>
        ) : processedFeatures.length > 0 ? (
          <div className="space-y-2">
            {processedFeatures.map((feature) => (
              <div key={feature.name} className="flex items-center gap-3">
                <span
                  className="text-xs text-gray-400 w-24 truncate"
                  title={feature.name}
                >
                  {feature.name}:
                </span>
                <div className="flex-1 h-2 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-[#39FF14] to-[#2acc0f] shadow-[0_0_10px_rgba(57,255,20,0.5)]"
                    style={{
                      width: `${(feature.value / maxImportance) * 100}%`,
                    }}
                  />
                </div>
                <span className="text-xs text-white font-semibold w-12 text-right">
                  {feature.value.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500">
            Train a model to see feature importance
          </p>
        )}
      </div>
    </div>
  );
}
