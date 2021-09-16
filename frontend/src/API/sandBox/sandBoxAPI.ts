import { instance } from "../baseURL";
import { IEpoch } from "./sandBoxInterface";

export const sandBoxAPI = {
  async getSandBoxEpoch(caseId: string) {
    try {
      const res = await instance.get<IEpoch[]>(`/sandbox/epoch/${caseId}`);
      return res.data;
    } catch (error: any) {
      console.error(`error`, error.response.data);
      return Promise.reject(error);
    }
  },

  async getSandBoxParams(caseId: string) {
    try {
      const params = await instance.get<any>(`/sandbox/params/${caseId}`);
      return params.data;
    } catch (error: any) {
      console.error(`error`, error.response.data);
      return Promise.reject(error);
    }
  },
};
