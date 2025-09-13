import axios from "axios";
import { API } from "../constants/api";

const api = axios.create({
  baseURL: "http://localhost/api",
});

export const evaluateCar = async () => {
  return await api.get(API.evaluate());
};
