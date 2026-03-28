export interface Environment {
  id: string;
  slug: string;
  name: string;
  description: string;
  tags: string[];
  fileCount?: number;
  totalSize?: number;
  readmePath?: string | null;
}

export interface EnvironmentFile {
  path: string;
  size: number;
  previewable: boolean;
}

export interface EnvironmentDetail {
  environment: Environment;
  files: EnvironmentFile[];
  readmePath: string | null;
  totalSize: number;
}
