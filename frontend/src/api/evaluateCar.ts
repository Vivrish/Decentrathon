import axios from "axios";
import { API } from "../constants/api";
import type { EvaluateCarResponse } from "@/types/evaluateCar";

const api = axios.create({
  baseURL: "http://localhost/api",
});

export const evaluateCar = async (
  files: File[]
): Promise<EvaluateCarResponse> => {
  const formData = new FormData();
  files.forEach((f) => formData.append("carImages", f));
  console.log(formData);
  return await api.post(API.evaluate(), formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
};
