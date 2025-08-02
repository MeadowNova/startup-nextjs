import ResumeOptimizer from "@/components/ResumeOptimizer";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Resume Optimizer",
  description: "Optimize your resume for ATS",
};

export default function OptimizePage() {
  return <ResumeOptimizer />;
}
