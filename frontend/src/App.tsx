import { evaluateCar } from "@/api/evaluateCar";
import "./App.css";
import { MultiImageUpload } from "./components/multiImageUpload";
import { Button } from "./components/ui/button";

async function uploadImages(files: File[]) {
  const res = await evaluateCar(files);
  console.log(res);
}

function App() {
  return (
    <>
      <div className="flex min-h-svh flex-col items-center justify-center">
        <MultiImageUpload onUpload={uploadImages} maxFiles={5} />
      </div>

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

export default App;
