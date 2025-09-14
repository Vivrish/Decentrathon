import { evaluateCar } from "@/api/evaluateCar";
import { MultiImageUpload } from "@/components/multiImageUpload";
import PageContainer from "@/components/PageContainer";
import { useNavigate } from "react-router-dom";

export default function UploadImagePage() {
  const navigate = useNavigate();

  async function uploadImages(files: File[]) {
    const res = (await evaluateCar(files)).data;
    console.log(res);
    navigate("/uploadImage/result", {
      state: {
        cleanliness: res.cleanliness,
        condition: res.condition,
        damagedCrops: res.damagedCrops,
        dirtyCrops: res.dirtyCrops,
      },
    });

    console.log(res.dirtyCrops);
  }

  return (
    <PageContainer>
      <MultiImageUpload onUpload={uploadImages} maxFiles={5} />
    </PageContainer>
  );
}
