import { ThunkAction } from "redux-thunk";
import { authAPI } from "../../../API/user/userAPI";

import { StateType } from "../../store";
import { IAuth } from "../../../API/user/authInterface";

const LOGIN_USER = "LOGIN_USER";
const LOGIN_USER_ERROR = "LOGIN_USER_ERROR";

let initialState = {
  user_info: null as IAuth | null,
  user_login_error: false,
};

export type InitialStateTypeAuthReduser = typeof initialState;
type AllTyps = LoginType | LoginErrorType;

const authReducer = (
  state = initialState,
  action: AllTyps
): InitialStateTypeAuthReduser => {
  switch (action.type) {
    case LOGIN_USER:
      return { ...state, user_info: action.data };
    case LOGIN_USER_ERROR:
      return { ...state, user_login_error: action.data };

    default:
      return state;
  }
};

type LoginType = {
  type: typeof LOGIN_USER;
  data: IAuth | null;
};
type LoginErrorType = {
  type: typeof LOGIN_USER_ERROR;
  data: boolean;
};

export const actionsAuth = {
  login: (data: IAuth | null): LoginType => ({ type: LOGIN_USER, data }),
  loginError: (data: boolean): LoginErrorType => ({
    type: LOGIN_USER_ERROR,
    data,
  }),
};

type ThunkTypeAsync = ThunkAction<Promise<void>, StateType, unknown, AllTyps>;
// type ThunkType = ThunkAction<void, StateType, unknown, AllTyps>;

export const authUser = (login: string, password: string): ThunkTypeAsync => {
  return async (dispatch) => {
    //   dispatch(actionsAuth.progress(true));
    dispatch(actionsAuth.loginError(false));
    //   dispatch(actionsAuth.isAuth(false));
    try {
      let data = await authAPI.signIn(login, password);
      // console.log("data", data);
      dispatch(actionsAuth.login(data));
      // dispatch(actionsAuth.progress(false));
      // dispatch(actionsAuth.isAuth(true));
      // dispatch(actionsAuth.JwtIsAuth(true));
    } catch (err) {
      // dispatch(actionsAuth.progress(false));
      dispatch(actionsAuth.loginError(true));
      console.log(err);
    }
  };
};

export default authReducer;
