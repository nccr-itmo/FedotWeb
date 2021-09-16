import { instance } from "../baseURL";
import { ICase, ICaseArr } from "./showCasesInterface";

export const showCasesAPI = {
  getShowCasesArr() {
    return instance
      .get<ICaseArr[]>(`/showcase/`)
      .then((res) => {
        // console.log("res.headers", res);
        return res.data;
      })
      .catch((error: any) => {
        console.error(`error`, error.response.data);
        return Promise.reject(error);
      });
  },
  getShowCaseById(caseId: string) {
    return instance
      .get<ICase>(`/showcase/items/${caseId}`)
      .then((res) => {
        return res.data;
      })
      .catch((error: any) => {
        // console.error(`error`, error.response.data);
        return Promise.reject(error);
      });
  },
};
