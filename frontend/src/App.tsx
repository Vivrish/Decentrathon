import { Button } from "@/components/ui/button";
import { evaluateCar } from "@/api/evaluateCar";
import "./App.css";

function App() {
  return (
    <>
      <div className="flex min-h-svh flex-col items-center justify-center">
        <Button onClick={evaluateCar}>Evaluate a car</Button>
      </div>
    </>
  );
}

export default App;
