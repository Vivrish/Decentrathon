// src/components/MultiImageUpload.tsx
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";

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
    <div className="flex flex-col gap-4 grow w-full items-center">
      <label
        className="flex w-full items-center justify-center px-4 py-8 
             bg-slate-100 text-gray-800 rounded-xl cursor-pointer 
             hover:bg-slate-200 
             transition-colors duration-300 ease-in-out"
      >
        <p className="font-medium text-lg">Выбрать фотографии +</p>
        <input
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileChange}
          disabled={loading}
          className="hidden"
        />
      </label>

      {error && <div className="text-red-500 text-sm">{error}</div>}
      <div className="space-y-3 grow">
        <div className="grid grid-cols-2 gap-3">
          {previewUrls.map((url, idx) => (
            <div
              key={idx}
              className="relative max-h-60 overflow-hidden rounded-lg"
            >
              <img
                src={url}
                alt={`preview-${idx}`}
                className="w-full object-center"
              />
              <button
                type="button"
                onClick={() => removeFile(idx)}
                className="absolute cursor-pointer top-2.5 right-2.5 bg-black/50 text-white rounded-full w-7 h-7 text-xs flex items-center justify-center"
              >
                <X className="size-5" />
              </button>
            </div>
          ))}
        </div>
      </div>
      <Button
        onClick={handleUpload}
        disabled={files.length === 0 || loading}
        variant="default"
      >
        {loading ? "Uploading…" : `Заугрзить ${files.length} фотографий`}
      </Button>
    </div>
  );
}
