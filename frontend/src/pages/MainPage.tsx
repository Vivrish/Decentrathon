import PageContainer from "@/components/PageContainer";
import { Button } from "../components/ui/button";
import { useNavigate } from "react-router-dom";
import AnimatedContent from "@/components/AnimatedContent";

export default function MainPage() {
  const navigate = useNavigate();

  return (
    <PageContainer>
      <div className="flex flex-col gap-2">
        <img src="/logos.png" alt="" className="w-55" />
        <p className="leading-5 text-sm text-neutral-500 text-center">
          Решение команды ExeQtion
        </p>
      </div>
      <div className="flex flex-col items-center justify-center grow gap-6">
        <div className="flex felx-col gap-2">
          <AnimatedContent
            distance={20}
            direction="vertical"
            reverse={false}
            duration={0.5}
            ease="power3.out"
            initialOpacity={0.0}
            animateOpacity
            scale={1.0}
            threshold={0.1}
            delay={0.0}
          >
            <img className="img-box" src="/images/main/Frame 6.png" />
          </AnimatedContent>
          <AnimatedContent
            distance={20}
            direction="vertical"
            reverse={false}
            duration={0.5}
            ease="power3.out"
            initialOpacity={0.0}
            animateOpacity
            scale={1.0}
            threshold={0.1}
            delay={0.1}
          >
            <img className="img-box" src="/images/main/Frame 7.png" />
          </AnimatedContent>
          <AnimatedContent
            distance={20}
            direction="vertical"
            reverse={false}
            duration={0.5}
            ease="power3.out"
            initialOpacity={0.0}
            animateOpacity
            scale={1.0}
            threshold={0.1}
            delay={0.3}
          >
            <img className="img-box" src="/images/main/Frame 8.png" />
          </AnimatedContent>
          <AnimatedContent
            distance={20}
            direction="vertical"
            reverse={false}
            duration={0.5}
            ease="power3.out"
            initialOpacity={0.0}
            animateOpacity
            scale={1.0}
            threshold={0.1}
            delay={0.4}
          >
            <img className="img-box" src="/images/main/Frame 9.png" />
          </AnimatedContent>
        </div>
        <div className="flex flex-col gap-5 items-center">
          <h1 className="text-center text-4xl font-extrabold tracking-tight text-balanced">
            Оцените состояние <mark className="highlight">автомобиля</mark>
          </h1>
          <p className="leading-5 text-neutral-500 max-w-70  text-center">
            Загрузите фотографию автомобиля и получите оценку его состояния
          </p>
        </div>
      </div>
      <Button onClick={() => navigate("/tips")}>Протестировать решение</Button>
    </PageContainer>
  );
}
