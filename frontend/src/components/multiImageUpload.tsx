// src/components/MultiImageUpload.tsx
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface MultiImageUploadProps {
  onUpload: (files: File[]) => Promise<void>;
  maxFiles?: number;
}

export function MultiImageUpload({
  onUpload,
  maxFiles = 5,
}: MultiImageUploadProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const selected = Array.from(e.target.files);

    if (files.length + selected.length > maxFiles) {
      setError(`You can only upload up to ${maxFiles} images.`);
      return;
    }

    setError(null);
    const newFiles = [...files, ...selected];
    setFiles(newFiles);

    const newUrls = selected.map((f) => URL.createObjectURL(f));
    setPreviewUrls((prev) => [...prev, ...newUrls]);
  };

  const removeFile = (index: number) => {
    const newFiles = files.filter((_, i) => i !== index);
    setFiles(newFiles);

    const newUrls = previewUrls.filter((_, i) => i !== index);
    setPreviewUrls(newUrls);
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    try {
      setLoading(true);
      await onUpload(files);
      setFiles([]);
      setPreviewUrls([]);
    } catch (err) {
      console.error(err);
      setError("Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-3">
        {previewUrls.map((url, idx) => (
          <div key={idx} className="relative w-32 h-32">
            <img
              src={url}
              alt={`preview-${idx}`}
              className="w-full h-full object-cover rounded-md"
            />
            <button
              type="button"
              onClick={() => removeFile(idx)}
              className="absolute top-1 right-1 bg-black/50 text-white rounded-full w-6 h-6 text-xs flex items-center justify-center"
            >
              ×
            </button>
          </div>
        ))}
      </div>

      <Input
        type="file"
        accept="image/*"
        multiple
        onChange={handleFileChange}
        disabled={loading}
      />

      {error && <div className="text-red-500 text-sm">{error}</div>}

      <Button
        onClick={handleUpload}
        disabled={files.length === 0 || loading}
        variant="default"
      >
        {loading ? "Uploading…" : `Upload ${files.length} image(s)`}
      </Button>
    </div>
  );
}
