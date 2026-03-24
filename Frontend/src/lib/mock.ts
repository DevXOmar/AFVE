import { v4fallback } from "./uuid";

export interface Evidence {
  source_url: string;
  snippet: string;
}

export interface Claim {
  claim: string;
  score: number;
  label: string;
  explanation: string;
  evidence: Evidence[];
}

export interface VerificationResult {
  overall_score: number;
  overall_label: string;
  claims: Claim[];
}

export function generateMockOutput(prompt: string): string {
  const lower = prompt.toLowerCase();
  if (lower.includes("einstein")) {
    return "Albert Einstein developed the theory of relativity in 1905. He was born in Ulm, Germany in 1879. He received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. Einstein also invented the lightbulb, which revolutionized modern society.";
  }
  if (lower.includes("moon")) {
    return "The Moon is Earth's only natural satellite. It orbits at an average distance of about 384,400 km. The first human landing was Apollo 11 in 1969, led by Neil Armstrong. The Moon has a thin atmosphere composed primarily of nitrogen.";
  }
  return "The topic you asked about involves several interconnected facts. Some claims are well-established in scientific literature, while others remain debated among experts. The key findings suggest a nuanced picture that requires careful evaluation of multiple evidence sources.";
}

export function generateMockVerification(prompt: string, modelOutput: string): VerificationResult {
  const lower = prompt.toLowerCase();

  if (lower.includes("einstein")) {
    return {
      overall_score: 0.72,
      overall_label: "Partially Factual",
      claims: [
        {
          claim: "Albert Einstein developed the theory of relativity in 1905",
          score: 0.95,
          label: "Verified",
          explanation: "Einstein published the special theory of relativity in 1905. The general theory came in 1915.",
          evidence: [
            { source_url: "https://en.wikipedia.org/wiki/Theory_of_relativity", snippet: "Einstein first published the special theory of relativity in 1905..." },
          ],
        },
        {
          claim: "He was born in Ulm, Germany in 1879",
          score: 0.99,
          label: "Verified",
          explanation: "This is a well-documented biographical fact.",
          evidence: [
            { source_url: "https://en.wikipedia.org/wiki/Albert_Einstein", snippet: "Albert Einstein was born on 14 March 1879 in Ulm..." },
          ],
        },
        {
          claim: "He received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect",
          score: 0.98,
          label: "Verified",
          explanation: "Correct. The Nobel Prize was awarded for his services to theoretical physics, especially the photoelectric effect.",
          evidence: [
            { source_url: "https://www.nobelprize.org/prizes/physics/1921/einstein/facts/", snippet: "The Nobel Prize in Physics 1921 was awarded to Albert Einstein..." },
          ],
        },
        {
          claim: "Einstein also invented the lightbulb",
          score: 0.02,
          label: "Hallucinated",
          explanation: "This is incorrect. Thomas Edison is widely credited with inventing the practical incandescent lightbulb.",
          evidence: [
            { source_url: "https://en.wikipedia.org/wiki/Incandescent_light_bulb", snippet: "Thomas Edison is credited with the invention of the first commercially practical incandescent light..." },
          ],
        },
      ],
    };
  }

  return {
    overall_score: 0.54,
    overall_label: "Uncertain",
    claims: [
      {
        claim: "The topic involves several interconnected facts",
        score: 0.6,
        label: "Uncertain",
        explanation: "This is a vague claim that cannot be precisely verified.",
        evidence: [],
      },
      {
        claim: "Some claims are well-established in scientific literature",
        score: 0.5,
        label: "Uncertain",
        explanation: "Without specific claims referenced, this is too general to verify.",
        evidence: [],
      },
    ],
  };
}

export function generateJobId(): string {
  return v4fallback();
}
