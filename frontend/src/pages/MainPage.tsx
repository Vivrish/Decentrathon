import { Button } from "../components/ui/button";

// async function uploadImages(files: File[]) {
//   const res = await evaluateCar(files);
//   console.log(res);
// }

export default function MainPage() {
  return (
    <div className="flex flex-col items-center space-between w-full h-lvh py-6">
      <div className="flex flex-col gap-2">
        <img src="/logos.png" alt="" className="w-55" />
        <p className="leading-5 text-sm text-neutral-500 text-center">
          Решение команды ExeQtion
        </p>
      </div>
      <div className="flex flex-col items-center justify-center grow gap-6">
        <div className="flex felx-col gap-2">
          <img
            className="w-18 h-24 bg-neutral-50 rounded-md"
            src="/images/main/Frame 6.png"
            alt=""
          />
          <img
            className="w-18 h-24 bg-neutral-50 rounded-md"
            src="/images/main/Frame 7.png"
            alt=""
          />
          <img
            className="w-18 h-24 bg-neutral-50 rounded-md"
            src="/images/main/Frame 8.png"
            alt=""
          />
          <img
            className="w-18 h-24 bg-neutral-50 rounded-md"
            src="/images/main/Frame 9.png"
            alt=""
          />
        </div>
        <div className="flex flex-col gap-5 items-center">
          <h1 className="text-center text-4xl font-extrabold tracking-tight text-balanced">
            Оцените состояние <mark className="highlight">автомобиля</mark>
          </h1>
          <p className="leading-5 text-neutral-500 max-w-70  text-center">
            Загрузите фотографий автомобиля и получите оценку безопасности
          </p>
          <Button className="max-w-90 w-full">Попробовать решение</Button>
        </div>
      </div>
      {/* <div>
        <Button className="bg-accent">Попробовать решение</Button>
      </div> */}
      {/* <div className="flex min-h-svh flex-col items-center justify-center">
        <MultiImageUpload onUpload={uploadImages} maxFiles={5} /> */}
      {/* </div> */}
    </div>
  );
}
