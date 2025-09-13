import { evaluateCar } from "@/api/evaluateCar";
import { MultiImageUpload } from "@/components/multiImageUpload";

async function uploadImages(files: File[]) {
  const res = await evaluateCar(files);
  console.log(res);
}

export default function UploadImagePage() {
  return (
    <div className="flex min-h-svh flex-col items-center justify-center">
      <MultiImageUpload onUpload={uploadImages} maxFiles={5} />
    </div>
  );
}
