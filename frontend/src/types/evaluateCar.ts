export type CarCleanliness = "Clean" | "Dirty";
export type CarCondition = "Good" | "Damaged";

export interface EvaluateCarResponse {
  cleanliness: CarCleanliness;
  condition: CarCondition;
  dirtyCrops: string[];
  damagedCrops: string[];
}
