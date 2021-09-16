import { instance } from "../baseURL";
import { IAuth } from "./authInterface";

export const authAPI = {
  signIn(email: string, password: string) {
    return instance
      .post<IAuth>(`/token/get_token`, { email: email, password: password })
      .then((res) => {
        // console.log("res.headers", res);
        return res.data;
      })
      .catch((error: any) => {
        console.error(`error`, error.response.data);
        return Promise.reject(error);
      });
  },
};
