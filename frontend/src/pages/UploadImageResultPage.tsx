import PageContainer from "@/components/PageContainer";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { PiggyBank, Sparkles, ThumbsUp, Wrench } from "lucide-react";
import { useNavigate } from "react-router-dom";

type CarStatusProps = {
  clean: boolean;
  intact: boolean;
};

const CarStatusAlert = ({ clean, intact }: CarStatusProps) => {
  const statuses = [
    {
      condition: clean,
      positive: {
        icon: <Sparkles className="mt-1" />,
        title: "Автомобиль чистый",
        description:
          "Чистый салон и кузов создают приятное впечатление и увеличивают шансы на высокую оценку. Отличная работа!",
        className: "bg-muted/75",
      },
      negative: {
        icon: <PiggyBank className="mt-1" />,
        title: "Автомобиль требует уборки",
        description:
          "Грязный салон и кузов портят впечатление и могут снизить оценки пассажиров. Рекомендуем выделить время на мойку.",
        className: "bg-destructive/50",
      },
    },
    {
      condition: intact,
      positive: {
        icon: <ThumbsUp className="mt-1 text-destructive" />,
        title: "Автомобиль целый",
        description:
          "Надёжная и целая машина — это не только комфорт, но и чувство безопасности для пассажиров. Так держать!",
        className: "bg-muted/75",
      },
      negative: {
        icon: <Wrench className="mt-1" />,
        title: "Автомобиль повреждён",
        description:
          "Повреждения кузова вызывают недовольство и снижают оценки пассажиров. Лучше обратиться в сервис для ремонта.",
        className: "bg-destructive/50",
      },
    },
  ];

  return (
    <>
      {statuses.map(({ condition, positive, negative }, i) => {
        const { icon, title, description, className } = condition
          ? positive
          : negative;
        return (
          <Alert key={i} className={`${className} border-none`}>
            {icon}
            <AlertTitle className="font-semibold text-lg">{title}</AlertTitle>
            <AlertDescription>{description}</AlertDescription>
          </Alert>
        );
      })}
    </>
  );
};

export default function UploadImageResultPage() {
  const navigate = useNavigate();
  return (
    <PageContainer>
      <h1 className="text-3xl leading-9 font-extrabold tracking-tight text-center text-balanced">
        Результат <mark className="highlight">оценки</mark>
      </h1>

      <div className="space-y-2 grow">
        {CarStatusAlert({ clean: true, intact: false })}

        <div className="grid grid-cols-2 gap-3">
          <img
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOJ-d07MI21CJ3wGgK-NYhpZlnrzeZWGJ3HA&s"
            className="max-h-60 overflow-hidden rounded-lg"
            alt=""
          />
          <img
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOJ-d07MI21CJ3wGgK-NYhpZlnrzeZWGJ3HA&s"
            className="max-h-60 overflow-hidden rounded-lg"
            alt=""
          />
          <img
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOJ-d07MI21CJ3wGgK-NYhpZlnrzeZWGJ3HA&s"
            className="max-h-60 overflow-hidden rounded-lg"
            alt=""
          />
          <img
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOJ-d07MI21CJ3wGgK-NYhpZlnrzeZWGJ3HA&s"
            className="max-h-60 overflow-hidden rounded-lg"
            alt=""
          />
        </div>
      </div>

      <Button onClick={() => navigate("/")}>Начать сначала</Button>
    </PageContainer>
  );
}
