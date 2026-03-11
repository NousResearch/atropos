import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { PassThrough } from "stream";
import archiver from "archiver";
import { getEnvById, getEnvDirPath, listEnvFiles } from "@/lib/env-api";

function nodeStreamToWeb(readable: NodeJS.ReadableStream): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      readable.on("data", (chunk: Buffer) => controller.enqueue(new Uint8Array(chunk)));
      readable.on("end", () => controller.close());
      readable.on("error", (err) => controller.error(err));
    },
  });
}

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const env = getEnvById(id);
  if (!env)
    return NextResponse.json({ error: "Environment not found" }, { status: 404 });
  const dirPath = getEnvDirPath(id);
  if (!dirPath)
    return NextResponse.json({ error: "Environment directory not found" }, { status: 404 });
  const files = listEnvFiles(id);
  const archive = archiver("zip", { zlib: { level: 5 } });
  const pass = new PassThrough();
  archive.pipe(pass);
  for (const rel of files) {
    const full = path.join(dirPath, rel);
    if (fs.statSync(full).isFile()) archive.file(full, { name: rel });
  }
  void archive.finalize();
  const filename = id.replace(/\//g, "-") + ".zip";
  return new NextResponse(nodeStreamToWeb(pass), {
    headers: {
      "Content-Type": "application/zip",
      "Content-Disposition": `attachment; filename="${filename}"`,
    },
  });
}
