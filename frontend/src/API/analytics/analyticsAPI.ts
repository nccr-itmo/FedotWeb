import { instance } from "../baseURL";
import { IGeneration, IMetric, IResult } from "./analyticsInterface";

export const analyticsAPI = {
  async getResults(caseId: string, pipelineId: string) {
    try {
      const res = await instance.get<IResult>(
        `/analytics/results/${caseId}/${pipelineId}`
      );
      // console.log(`res`, res.data);
      return res.data;
    } catch (error: any) {
      console.error(`error`, error.response.data);
      return Promise.reject(error);
    }
  },
  async getMetrics(caseId: string) {
    try {
      const res = await instance.get<IMetric>(`/analytics/quality/${caseId}`);
      // console.log(`res`, res.data);
      return res.data;
    } catch (error: any) {
      console.error(`error`, error.response.data);
      return Promise.reject(error);
    }
  },
  async getGenerations(caseId: string, type: string) {
    try {
      const res = await instance.get<IGeneration>(
          `/analytics/generations/${caseId}/${type}`
      );
      return res.data;
    } catch (error: any) {
      console.error(`error`, error.response.data);
      return Promise.reject(error);
    }
  },
};
