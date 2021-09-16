import { ThunkAction } from "redux-thunk";
import { analyticsAPI } from "../../../API/analytics/analyticsAPI";
import { IGeneration } from "../../../API/analytics/analyticsInterface";
import { StateType } from "../../store";

const HISTORY_GENERATIONS_PHENO = "HISTORY_GENERATIONS_PHENO";
const HISTORY_GENERATIONS_GENO = "HISTORY_GENERATIONS_GENO";
const HISTORY_GENOTHYPIC_PHENOTHYPIC = "HISTORY_GENOTHYPIC_PHENOTHYPIC";

let initialState = {
  history_generations_pheno: null as IGeneration | null,
  history_generations_geno: null as IGeneration | null,
  history_genothypic_phenothypic: true,
};

export type InitialStateType = typeof initialState;
type AllTyps =
  | GenerationsPhenoType
  | GenerationsGenoType
  | GenothypicPhenothypicType;

const historyReducer = (
  state = initialState,
  action: AllTyps
): InitialStateType => {
  switch (action.type) {
    case HISTORY_GENERATIONS_PHENO:
      return { ...state, history_generations_pheno: action.data };
    case HISTORY_GENERATIONS_GENO:
      return { ...state, history_generations_geno: action.data };
    case HISTORY_GENOTHYPIC_PHENOTHYPIC:
      return {
        ...state,
        history_genothypic_phenothypic: action.data,
      };

    default:
      return state;
  }
};

type GenerationsPhenoType = {
  type: typeof HISTORY_GENERATIONS_PHENO;
  data: IGeneration | null;
};
type GenerationsGenoType = {
  type: typeof HISTORY_GENERATIONS_GENO;
  data: IGeneration | null;
};
type GenothypicPhenothypicType = {
  type: typeof HISTORY_GENOTHYPIC_PHENOTHYPIC;
  data: boolean;
};

export const actionsHistory = {
  generationsPheno: (data: IGeneration | null): GenerationsPhenoType => ({
    type: HISTORY_GENERATIONS_PHENO,
    data,
  }),
  generationsGeno: (data: IGeneration | null): GenerationsGenoType => ({
    type: HISTORY_GENERATIONS_GENO,
    data,
  }),
  genothypicPhenothypic: (data: boolean): GenothypicPhenothypicType => ({
    type: HISTORY_GENOTHYPIC_PHENOTHYPIC,
    data,
  }),
};

type ThunkTypeAsync = ThunkAction<Promise<void>, StateType, unknown, AllTyps>;
type ThunkType = ThunkAction<void, StateType, unknown, AllTyps>;

export const getGenerations = (
  caseId: string,
  type: string
): ThunkTypeAsync => {
  return async (dispatch) => {
    try {
      let generations = await analyticsAPI.getGenerations(caseId, type);
      if (type === "pheno")
        dispatch(actionsHistory.generationsPheno(generations));
      else if (type === "geno")
        dispatch(actionsHistory.generationsGeno(generations));
    } catch (err) {
      console.error(err);
    }
  };
};

export const setGenerations = (
  generations: IGeneration | null,
  type: string
): ThunkType => {
  return (dispatch) => {
    if (type === "pheno")
      dispatch(actionsHistory.generationsPheno(generations));
    else if (type === "geno")
      dispatch(actionsHistory.generationsGeno(generations));
  };
};

export const setgenothypicPhenothypic = (data: boolean): ThunkType => {
  return (dispatch) => {
    dispatch(actionsHistory.genothypicPhenothypic(data));
  };
};

export default historyReducer;
