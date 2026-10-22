"use client";

import { useParams } from "next/navigation";
import { ProjectView } from "@/components/ProjectView";

export default function ProjectPage() {
  const params = useParams<{ id: string }>();
  return <ProjectView id={params.id} />;
}
