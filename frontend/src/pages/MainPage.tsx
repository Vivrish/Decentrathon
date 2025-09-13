import { Button } from "../components/ui/button";

export default function MainPage() {
  return (
    <>
      <div className="flex items-center flex-col">
        <div className="flex min-w-[370px] flex-col items-center justify-center bg-red-100">
          <p>Решение команды ExeQtion</p>
          <h1>
            Оцените состояние <mark className="bg-amber-100">автомобиля</mark>
          </h1>
          <p>Загрузите фотографий автомобиля и получите оценку безопасности</p>
          <Button>Оценить состояние</Button>
        </div>
      </div>
    </>
  );
}
