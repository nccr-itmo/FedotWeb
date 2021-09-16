import { ThunkAction } from "redux-thunk";
import { sandBoxAPI } from "../../../API/sandBox/sandBoxAPI";
import { StateType } from "../../store";
import { IEpoch } from "./../../../API/sandBox/sandBoxInterface";
import { analyticsAPI } from "./../../../API/analytics/analyticsAPI";
import { IMetric, IResult } from "../../../API/analytics/analyticsInterface";
import {
  EdgeDataType,
  NodeDataType,
} from "../../../workSpace/pages/sandbox/GraphEditor/GraphEditorDirectedGraph/GraphEditorDirectedGraph";
import { pipelineAPI } from "../../../API/pipeline/pipelineAPI";
import { getMainGraph } from "./sandbox-reducer";

interface Iedit_or_add_form_opened {
  isOpen: boolean;
  type: "new" | "edit";
}

const SANDBOX_EPOCHS = "SANDBOX_EPOCHS";
const SANDBOX_EPOCH_SELCTED = "SANDBOX_EPOCH_SELCTED";
const SANDBOX_RESULTS = "SANDBOX_RESULTS";
const SANDBOX_METRICS = "SANDBOX_METRICS";
const SANDBOX_HISTORY = "SANDBOX_HISTORY";
const SANDBOX_EDIT_OR_ADD_FORM_OPENED = "SANDBOX_EDIT_OR_ADD_FORM_OPENED";
const SANDBOX_INITIAL_VALUES_EDIT_POINT = "SANDBOX_INITIAL_VALUES_EDIT_POINT";
const SANDBOX_PARAMS = "SANDBOX_PARAMS";

let initialState = {
  sandBox_epochs: null as IEpoch[] | null,
  sandBox_epochs_selected: null as IEpoch | null,
  sandbox_results: null as IResult | null,
  sandbox_metrics: null as IMetric | null,
  sandbox_history: false,
  sandbox_edit_or_add_form_opened: {
    isOpen: false,
    type: "new",
  } as Iedit_or_add_form_opened,
  sandbox_Initial_values_edit_piont: null as null | NodeDataType,
  sandbox_params: null as null | any,
};

export type InitialStateType = typeof initialState;
type AllTyps =
  | ParamsType
  | InitialValuesEditFormType
  | EditOrAddFormOpenedType
  | EpochsArrType
  | EpochsSelectedType
  | ResultsType
  | MetricsType
  | HistipyType;

const sandBoxReducer = (
  state = initialState,
  action: AllTyps
): InitialStateType => {
  switch (action.type) {
    case SANDBOX_EPOCHS:
      return { ...state, sandBox_epochs: action.data };
    case SANDBOX_EPOCH_SELCTED:
      return { ...state, sandBox_epochs_selected: action.data };
    case SANDBOX_RESULTS:
      return { ...state, sandbox_results: action.data };
    case SANDBOX_METRICS:
      return { ...state, sandbox_metrics: action.data };
    case SANDBOX_HISTORY:
      return { ...state, sandbox_history: action.data };
    case SANDBOX_EDIT_OR_ADD_FORM_OPENED:
      return { ...state, sandbox_edit_or_add_form_opened: action.data };
    case SANDBOX_INITIAL_VALUES_EDIT_POINT:
      return { ...state, sandbox_Initial_values_edit_piont: action.data };
    case SANDBOX_PARAMS:
      return { ...state, sandbox_params: action.data };

    default:
      return state;
  }
};

type EpochsArrType = {
  type: typeof SANDBOX_EPOCHS;
  data: IEpoch[] | null;
};
type EpochsSelectedType = {
  type: typeof SANDBOX_EPOCH_SELCTED;
  data: IEpoch | null;
};
type ResultsType = {
  type: typeof SANDBOX_RESULTS;
  data: IResult | null;
};
type MetricsType = {
  type: typeof SANDBOX_METRICS;
  data: IMetric | null;
};
type HistipyType = {
  type: typeof SANDBOX_HISTORY;
  data: boolean;
};
type EditOrAddFormOpenedType = {
  type: typeof SANDBOX_EDIT_OR_ADD_FORM_OPENED;
  data: Iedit_or_add_form_opened;
};
type InitialValuesEditFormType = {
  type: typeof SANDBOX_INITIAL_VALUES_EDIT_POINT;
  data: null | NodeDataType;
};
type ParamsType = {
  type: typeof SANDBOX_PARAMS;
  data: null | any;
};

export const actionsSandBox = {
  epochsArr: (data: IEpoch[] | null): EpochsArrType => ({
    type: SANDBOX_EPOCHS,
    data,
  }),
  epochsSelected: (data: IEpoch | null): EpochsSelectedType => ({
    type: SANDBOX_EPOCH_SELCTED,
    data,
  }),
  results: (data: IResult | null): ResultsType => ({
    type: SANDBOX_RESULTS,
    data,
  }),
  metrics: (data: IMetric | null): MetricsType => ({
    type: SANDBOX_METRICS,
    data,
  }),
  historyToggle: (data: boolean): HistipyType => ({
    type: SANDBOX_HISTORY,
    data,
  }),
  setEditOrAddFormOpened: (
    data: Iedit_or_add_form_opened
  ): EditOrAddFormOpenedType => ({
    type: SANDBOX_EDIT_OR_ADD_FORM_OPENED,
    data,
  }),
  setInitialValuesEditForm: (
    data: null | NodeDataType
  ): InitialValuesEditFormType => ({
    type: SANDBOX_INITIAL_VALUES_EDIT_POINT,
    data,
  }),
  setParams: (data: null | any): ParamsType => ({
    type: SANDBOX_PARAMS,
    data,
  }),
};

type ThunkTypeAsync = ThunkAction<Promise<void>, StateType, unknown, AllTyps>;
type ThunkType = ThunkAction<void, StateType, unknown, AllTyps>;

export const getEpochs = (caseId: string): ThunkTypeAsync => {
  return async (dispatch) => {
    try {
      let epochs = await sandBoxAPI.getSandBoxEpoch(caseId);
      dispatch(actionsSandBox.epochsArr(epochs));
    } catch (err) {
      console.error(err);
    }
  };
};

export const setSelectedEpoch = (epoch: IEpoch): ThunkType => {
  return (dispatch) => {
    dispatch(actionsSandBox.epochsSelected(epoch));
  };
};

export const getResults = (
  caseId: string,
  pipelineId: string
): ThunkTypeAsync => {
  // console.log(`caseId`, caseId);
  return async (dispatch) => {
    dispatch(actionsSandBox.results(null));
    try {
      let results = await analyticsAPI.getResults(caseId, pipelineId);
      dispatch(actionsSandBox.results(results));
    } catch (err) {
      console.error(err);
    }
  };
};

export const getMetrics = (caseId: string): ThunkTypeAsync => {
  return async (dispatch) => {
    try {
      let metrics = await analyticsAPI.getMetrics(caseId);
      dispatch(actionsSandBox.metrics(metrics));
    } catch (err) {
      console.error(err);
    }
  };
};

export const setHistoryToggle = (data: boolean): ThunkType => {
  return (dispatch) => {
    dispatch(actionsSandBox.historyToggle(data));
  };
};

export const setEditOrAddFormOpened = (
  data: Iedit_or_add_form_opened
): ThunkType => {
  return (dispatch) => {
    dispatch(actionsSandBox.setEditOrAddFormOpened(data));
  };
};

export const setInitialValuesEditForm = (
  node: NodeDataType | undefined | null
): ThunkType => {
  return (dispatch) => {
    if (node) {
      dispatch(
        actionsSandBox.setEditOrAddFormOpened({ isOpen: false, type: "new" })
      );
      dispatch(actionsSandBox.setInitialValuesEditForm(node));
    } else {
      dispatch(actionsSandBox.setInitialValuesEditForm(null));
    }
  };
};

export const getParams = (caseId: string): ThunkTypeAsync => {
  return async (dispatch) => {
    try {
      let params = await sandBoxAPI.getSandBoxParams(caseId);
      dispatch(actionsSandBox.setParams(params));
    } catch (err) {
      console.error(err);
    }
  };
};

export const sendNewGraphOnServer = (
  caseId: string,
  uid: string,
  nodes: NodeDataType[],
  edges: EdgeDataType[]
): ThunkTypeAsync => {
  return async (dispatch) => {
    try {
      let validate = await pipelineAPI.postValidate(uid, nodes, edges);
      if (validate.is_valid) {
        let editedGraph = await pipelineAPI.postAdd(uid, nodes, edges);
        if (editedGraph.is_new) {
          dispatch(getMainGraph(editedGraph.uid));
          let results = await analyticsAPI.getResults(caseId, editedGraph.uid);
          dispatch(actionsSandBox.results(results));
           alert('Graph is valid. Added.');
        }
      } else {
        alert(validate.error_desc);
      }
    } catch (err) {
      console.error(err);
    }
  };
};

export default sandBoxReducer;
