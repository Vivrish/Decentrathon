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

      <div className="grow flex flex-col justify-center gap-8">
        <div className="flex gap-6 flex-col">
          <div className="flex gap-2">
            <img className="size-5 mt-1" src="/images/icons/check.svg" alt="" />
            <div className="flex flex-col gap-1">
              <h3 className="text-xl font-semibold tracking-tight">
                Загрузите одну или больше фотографий
              </h3>
              <p className="leading-5 text-sm text-neutral-500 max-w-100">
                Cфотографируйте автомобиль днём, полностью, под углом, без
                посторонних объектов
              </p>
            </div>
          </div>
          <div className="flex felx-col gap-2">
            <img className="img-box" src="/images/correct/Frame 9.png" />
            <img className="img-box" src="/images/correct/Frame 35.png" />
            <img className="img-box" src="/images/correct/Frame 36.png" />
            <img className="img-box" src="/images/correct/Frame 37.png" />
          </div>
        </div>
        <Separator />
        <div className="flex gap-6 flex-col">
          <div className="flex gap-2">
            <img className="size-5 mt-1" src="/images/icons/x.svg" alt="" />
            <div className="flex flex-col gap-1">
              <h3 className="text-xl font-semibold tracking-tight">
                Не загружайте фотографии
              </h3>
              <p className="leading-5 text-sm text-neutral-500 max-w-100">
                Темные или против солнца, размытые, обрезанные с посторонними
                объектами, с фильтрами или обработкой
              </p>
            </div>
          </div>
          <div className="flex felx-col gap-2">
            <img className="img-box" src="/images/incorrect/Frame 38.png" />
            <img className="img-box" src="/images/incorrect/Frame 39.png" />
            <img className="img-box" src="/images/incorrect/Frame 40.png" />
            <img className="img-box" src="/images/incorrect/Frame 41.png" />
          </div>
        </div>
      </div>

      <Button onClick={() => navigate("/uploadImage")}>
        Загрузить фотографии
      </Button>
    </PageContainer>
  );
}
