import { NextRequest, NextResponse } from "next/server";
import { getEnvById, listEnvFiles, listEnvFilesWithSizes } from "@/lib/env-api";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  if (!getEnvById(id))
    return NextResponse.json({ error: "Environment not found" }, { status: 404 });
  const withSizes = request.nextUrl.searchParams.get("sizes") === "1";
  if (withSizes) {
    const files = listEnvFilesWithSizes(id);
    return NextResponse.json(files);
  }
  const files = listEnvFiles(id);
  return NextResponse.json(files);
}
