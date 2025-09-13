import type { ReactNode } from "react";

interface PageContainerProps {
  children: ReactNode;
}

export default function PageContainer({ children }: PageContainerProps) {
  return (
    <div className="flex flex-col items-center min-h-screen">
      <div className="flex flex-col items-center space-between grow w-full max-w-[470px] p-6 bg-white rounded-xl">
        {children}
      </div>
    </div>
  );
}
