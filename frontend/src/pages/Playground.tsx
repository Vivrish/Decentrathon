import PageContainer from "@/components/PageContainer";
import { Button } from "@/components/ui/button";
import type { CarCleanliness, CarCondition } from "@/types/evaluateCar";
import { useLocation, useNavigate } from "react-router-dom";

type CarStatusProps = {
  clean: boolean;
  intact: boolean;
};

const CarStatusAlert = ({ clean, intact }: CarStatusProps) => {
  const statuses = [
    {
      condition: clean,
      positive: {
        icon: "/images/results/stars.png",
        title: "Автомобиль чистый",
        description:
          "Чистый салон и кузов создают приятное впечатление и увеличивают шансы на высокую оценку. Отличная работа!",
      },
      negative: {
        icon: "/images/results/pig.png",
        title: "Автомобиль требует уборки",
        description:
          "Грязный салон и кузов портят впечатление и могут снизить оценки пассажиров. Рекомендуем выделить время на мойку.",
      },
    },
    {
      condition: intact,
      positive: {
        icon: "/images/results/thumb-up.png",
        title: "Автомобиль целый",
        description:
          "Надёжная и целая машина — это не только комфорт, но и чувство безопасности для пассажиров. Так держать!",
        className: "bg-muted/75",
      },
      negative: {
        icon: "/images/results/wrench.png",
        title: "Автомобиль повреждён",
        description:
          "Повреждения кузова вызывают недовольство и снижают оценки пассажиров. Лучше обратиться в сервис для ремонта.",
      },
    },
  ];

  return (
    <>
      {statuses.map(({ condition, positive, negative }, i) => {
        const { icon, title, description } = condition ? positive : negative;
        return (
          <div key={i} className={`border-none flex gap-3`}>
            <img src={icon} className="size-12 mt-1 rounded-xl" alt="" />
            <div>
              <h3 className="font-bold text-lg">{title}</h3>
              <p className="leading-5 text-sm text-neutral-500 max-w-100">
                {description}
              </p>
            </div>
          </div>
        );
      })}
    </>
  );
};

export default function UploadImageResultPage() {
  const { state } = useLocation();
  const navigate = useNavigate();

  // Provide fallback/mock data if state is undefined
  const cleanliness: CarCleanliness = state?.cleanliness ?? "Clean";
  const condition: CarCondition = state?.condition ?? "Good";
  const damagedCrops: string[] = state?.damagedCrops ?? [];
  const dirtyCrops: string[] = state?.dirtyCrops ?? [];

  return (
    <PageContainer>
      <h1 className="text-3xl leading-9 font-extrabold tracking-tight text-center text-balanced">
        Результат <mark className="highlight">оценки</mark>
      </h1>

      <div className="space-y-6 grow mt-8">
        <CarStatusAlert
          clean={cleanliness === "Clean"}
          intact={condition === "Good"}
        />

        <div className="bg-slate-50 p-4 flex flex-col gap-2 rounded-xl">
          <h3 className="font-bold text-lg">Проблемные зоны</h3>
          <div className="grid grid-cols-2 gap-3">
            {dirtyCrops.map((img, i) => (
              <img
                key={`dirty-${i}`}
                src={`data:image/png;base64,${img}`}
                className="max-h-60 overflow-hidden rounded-lg"
                alt={`dirty crop ${i + 1}`}
              />
            ))}

            {damagedCrops.map((img, i) => (
              <img
                key={`damaged-${i}`}
                src={`data:image/png;base64,${img}`}
                className="max-h-60 overflow-hidden rounded-lg"
                alt={`damaged crop ${i + 1}`}
              />
            ))}
          </div>
        </div>
      </div>

      <Button onClick={() => navigate("/")}>Начать сначала</Button>
    </PageContainer>
  );
}
