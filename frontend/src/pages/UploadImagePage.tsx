import { evaluateCar } from "@/api/evaluateCar";
import { MultiImageUpload } from "@/components/multiImageUpload";
import PageContainer from "@/components/PageContainer";

async function uploadImages(files: File[]) {
  const res = await evaluateCar(files);
  console.log(res);
}

export default function UploadImagePage() {
  return (
    <PageContainer>
      <MultiImageUpload onUpload={uploadImages} maxFiles={5} />
    </PageContainer>
  );
}
