import { ThunkAction } from "redux-thunk";
import { StateType } from "../../store";
import { ICase, ICaseArr } from "../../../API/showCase/showCasesInterface";
import { showCasesAPI } from "../../../API/showCase/showCasesAPI";

const SHOW_CASES_ARR = "SHOW_CASES_ARR";
const SHOW_CASE_BY_ID = "SHOW_CASE_BY_ID";

let initialState = {
  show_cases_arr: null as ICaseArr[] | null,
  show_case_by_id: null as ICase | null,
};

export type InitialStateType = typeof initialState;
type AllTyps = CasesArrType | CaseByIdType;

const casesReducer = (
  state = initialState,
  action: AllTyps
): InitialStateType => {
  switch (action.type) {
    case SHOW_CASES_ARR:
      return { ...state, show_cases_arr: action.data };
    case SHOW_CASE_BY_ID:
      return { ...state, show_case_by_id: action.data };

    default:
      return state;
  }
};

type CasesArrType = {
  type: typeof SHOW_CASES_ARR;
  data: ICaseArr[] | null;
};
type CaseByIdType = {
  type: typeof SHOW_CASE_BY_ID;
  data: ICase | null;
};

export const actionsCases = {
  showCasesArr: (data: ICaseArr[] | null): CasesArrType => ({
    type: SHOW_CASES_ARR,
    data,
  }),
  showCaseById: (data: ICase | null): CaseByIdType => ({
    type: SHOW_CASE_BY_ID,
    data,
  }),
};

export const getShowCasesArr = (): ThunkTypeAsync => {
  return async (dispatch) => {
    try {
      let showCases = await showCasesAPI.getShowCasesArr();
      dispatch(actionsCases.showCasesArr(showCases));
    } catch (err) {
      console.error(err);
    }
  };
};

export const getShowCaseById = (caseId: string | null): ThunkTypeAsync => {
  return async (dispatch) => {
    dispatch(actionsCases.showCaseById(null));
    if (caseId !== null) {
      try {
        let caseById = await showCasesAPI.getShowCaseById(caseId);
        dispatch(actionsCases.showCaseById(caseById));
      } catch (err) {
        console.error(err);
      }
    } else {
      dispatch(actionsCases.showCaseById(null));
    }
  };
};

type ThunkTypeAsync = ThunkAction<Promise<void>, StateType, unknown, AllTyps>;
// type ThunkType = ThunkAction<void, StateType, unknown, AllTyps>;

export default casesReducer;
