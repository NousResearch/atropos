import { EnvironmentDetailClient } from "@/components/EnvironmentDetailClient";
import { readStaticManifest } from "@/lib/env-static";

export const dynamicParams = false;

export function generateStaticParams() {
  return readStaticManifest().map((environment) => ({
    id: environment.slug,
  }));
}

export default function EnvironmentDetailPage({
  params,
}: {
  params: { id: string };
}) {
  return <EnvironmentDetailClient slug={decodeURIComponent(params.id)} />;
}
