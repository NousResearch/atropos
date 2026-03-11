import { NextRequest, NextResponse } from "next/server";
import { getEnvById } from "@/lib/env-api";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const env = getEnvById(id);
  if (!env) return NextResponse.json({ error: "Not found" }, { status: 404 });
  return NextResponse.json(env);
}
