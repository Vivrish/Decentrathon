import PageContainer from "@/components/PageContainer";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

import { useNavigate } from "react-router-dom";

export default function TipsPage() {
  const navigate = useNavigate();

  return (
    <PageContainer>
      <h1 className="text-3xl leading-9 font-extrabold tracking-tight text-center text-balanced">
        Советы для лучших <br />
        <mark className="highlight">результатов</mark>
      </h1>

      <div className=" flex flex-col justify-center gap-8 mt-6">
        <div className="flex gap-3">
          <img className="size-5 mt-1" src="/images/icons/check.svg" alt="" />
          <div className="flex flex-col gap-4">
            <h3 className="text-xl font-semibold tracking-tight">
              Загрузите одну или больше фотографий вашего автомобиля
            </h3>
            <p className="leading-5 text-sm text-neutral-500 max-w-100">
              Снимайте в хорошем освещении, чтобы автомобиль был полностью
              виден, с разных ракурсов и без посторонних объектов.
            </p>
            <div className="flex felx-col gap-2">
              <img
                className="img-box-small"
                src="/images/correct/Frame 9.png"
              />
              <img
                className="img-box-small"
                src="/images/correct/Frame 35.png"
              />
              <img
                className="img-box-small"
                src="/images/correct/Frame 36.png"
              />
              <img
                className="img-box-small"
                src="/images/correct/Frame 37.png"
              />
            </div>
          </div>
        </div>
        <Separator />
        <div className="flex gap-3">
          <img className="size-5 mt-1" src="/images/icons/x.svg" alt="" />
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <h3 className="text-xl font-semibold tracking-tight">
                Какие фотографии не стоит загружать
              </h3>
              <p className="leading-5 text-sm text-neutral-500 max-w-100">
                Не загружайте тёмные, размытые, обработанные фильтрами, сильно
                обрезанные фотографии, с посторонними объектами.
              </p>
            </div>
            <div className="flex felx-col gap-2">
              <img
                className="img-box-small"
                src="/images/incorrect/Frame 38.png"
              />
              <img
                className="img-box-small"
                src="/images/incorrect/Frame 39.png"
              />
              <img
                className="img-box-small"
                src="/images/incorrect/Frame 40.png"
              />
              <img
                className="img-box-small"
                src="/images/incorrect/Frame 41.png"
              />
            </div>
          </div>
        </div>
      </div>

      <Button className="mt-8" onClick={() => navigate("/uploadImage")}>
        Загрузить фотографии
      </Button>
    </PageContainer>
  );
}
