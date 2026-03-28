import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import { getEnvById, getEnvFilePath } from "@/lib/env-api";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string; path: string[] }> }
) {
  const { id, path: pathSegments } = await params;
  if (!getEnvById(id))
    return NextResponse.json({ error: "Environment not found" }, { status: 404 });
  const filePath = pathSegments.join("/");
  const fullPath = getEnvFilePath(id, filePath);
  if (!fullPath)
    return NextResponse.json({ error: "File not found" }, { status: 404 });
  const stat = fs.statSync(fullPath);
  const stream = fs.createReadStream(fullPath);
  const res = new NextResponse(stream as unknown as ReadableStream);
  res.headers.set("Content-Length", String(stat.size));
  const ext = fullPath.split(".").pop()?.toLowerCase();
  const types: Record<string, string> = {
    py: "text/x-python",
    md: "text/markdown",
    json: "application/json",
    yaml: "text/yaml",
    yml: "text/yaml",
    toml: "text/plain",
    txt: "text/plain",
  };
  if (ext && types[ext]) res.headers.set("Content-Type", types[ext]);
  return res;
}
