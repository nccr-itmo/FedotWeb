import { StateType } from "../../store";
import { ThunkAction } from "redux-thunk";

import { IHistoryGraph, IMainGraph, sandboxAPI } from "../../../API/sandbox";
import {
  EdgeDataType,
  edgeValueType,
  NodeDataType,
} from "../../../workSpace/pages/sandbox/GraphEditor/GraphEditorDirectedGraph/GraphEditorDirectedGraph";

const SET_MAIN_GRAPH = "SET_MAIN_GRAPH";
const SET_HISTORY_GRAPH = "SET_HISTORY_GRAPH";
const DELETE_NODE_MAIN_GRAPH = "DELETE_NODE_MAIN_GRAPH";
const ADD_NODE_MAIN_GRAPH = "ADD_NODE_MAIN_GRAPH";
const TOGGLE_EDIT_MODAL = "TOGGLE_EDIT_MODAL";
const DELETE_EDGE_MAIN_GRAPH = "DELETE_EDGE_MAIN_GRAPH";
const ADD_EDGE_MAIN_GRAPH = "ADD_EDGE_MAIN_GRAPH";

let initialState = {
  mainGraph: { nodes: [], edges: [] } as IMainGraph,
  historyGraph: { nodes: [], edges: [] } as IHistoryGraph,
  isOpenEditModal: false,
};

export type InitialStateDatasetsType = typeof initialState;
type AllTypes =
  | SetMainGraphType
  | SetHistoryGraphType
  | DeleteNodeMainGraphType
  | AddNodeMainGraphType
  | ToggleEditModalType
  | DeleteEdgeMainGraphType
  | AddEdgeMainGraphType;

const sandbox_Egor = (
  state = initialState,
  action: AllTypes
): InitialStateDatasetsType => {
  switch (action.type) {
    case SET_MAIN_GRAPH:
      return { ...state, mainGraph: action.data };
    case SET_HISTORY_GRAPH:
      return {
        ...state,
        historyGraph: action.data,
      };
    case DELETE_NODE_MAIN_GRAPH: {
      return {
        ...state,
        mainGraph: {
          ...state.mainGraph,
          nodes: [...state.mainGraph.nodes].filter((item) => {
            return item.id !== +action.data;
          }),
          edges: [...state.mainGraph.edges].filter((item) => {
            return item.source !== +action.data && item.target !== +action.data;
          }),
        },
      };
    }
    case ADD_NODE_MAIN_GRAPH: {
      // console.log(`### action.data`, action.data);
      return {
        ...state,
        mainGraph: {
          ...state.mainGraph,
          nodes: [...state.mainGraph.nodes, ...action.data.nodes],
          edges: [...state.mainGraph.edges, ...action.data.edges],
        },
      };
    }
    case TOGGLE_EDIT_MODAL: {
      return {
        ...state,
        isOpenEditModal: !state.isOpenEditModal,
      };
    }
    case DELETE_EDGE_MAIN_GRAPH: {
      return {
        ...state,
        mainGraph: {
          ...state.mainGraph,
          edges: [...state.mainGraph.edges].filter((item) => {
            // console.log(
            //   `### item.source, action.data.v`,
            //   String(item.source) !== action.data.v
            // );
            // console.log(
            //   `### item.target, action.data.w`,
            //   String(item.target) !== action.data.w
            // );
            // console.log(
            //   `return`,
            //   String(item.source) !== action.data.v ||
            //     String(item.target) !== action.data.w
            // );
            return (
              String(item.source) !== action.data.v ||
              String(item.target) !== action.data.w
            );
          }),
        },
      };
    }
    case ADD_EDGE_MAIN_GRAPH: {
      const hadEdge = state.mainGraph.edges.find(
        (item) =>
          item.source === action.data.source &&
          item.target === action.data.target
      );
      if (hadEdge) {
        return { ...state };
      }

      return {
        ...state,
        mainGraph: {
          ...state.mainGraph,
          edges: [...state.mainGraph.edges, action.data],
        },
      };
    }
    default:
      return state;
  }
};

type SetMainGraphType = {
  type: typeof SET_MAIN_GRAPH;
  data: IMainGraph;
};
type SetHistoryGraphType = {
  type: typeof SET_HISTORY_GRAPH;
  data: IHistoryGraph;
};
type DeleteNodeMainGraphType = {
  type: typeof DELETE_NODE_MAIN_GRAPH;
  data: number;
};
type AddNodeMainGraphType = {
  type: typeof ADD_NODE_MAIN_GRAPH;
  data: IMainGraph;
};
type ToggleEditModalType = {
  type: typeof TOGGLE_EDIT_MODAL;
};
type DeleteEdgeMainGraphType = {
  type: typeof DELETE_EDGE_MAIN_GRAPH;
  data: edgeValueType;
};
type AddEdgeMainGraphType = {
  type: typeof ADD_EDGE_MAIN_GRAPH;
  data: EdgeDataType;
};

export const actionsSandbox = {
  setMainGraph: (data: IMainGraph): SetMainGraphType => ({
    type: SET_MAIN_GRAPH,
    data,
  }),
  setHistoryGraph: (data: IHistoryGraph): SetHistoryGraphType => ({
    type: SET_HISTORY_GRAPH,
    data,
  }),
  deleteNodeMainGraph: (data: number): DeleteNodeMainGraphType => ({
    type: DELETE_NODE_MAIN_GRAPH,
    data,
  }),
  addNodeMainGraph: (data: IMainGraph): AddNodeMainGraphType => ({
    type: ADD_NODE_MAIN_GRAPH,
    data,
  }),
  toggleEditModal: (): ToggleEditModalType => ({
    type: TOGGLE_EDIT_MODAL,
  }),
  deleteEdgeMainGraph: (data: edgeValueType): DeleteEdgeMainGraphType => ({
    type: DELETE_EDGE_MAIN_GRAPH,
    data,
  }),
  addEdgeMainGraph: (data: EdgeDataType): AddEdgeMainGraphType => ({
    type: ADD_EDGE_MAIN_GRAPH,
    data,
  }),
};
// для async
type ThunkTypeAsync = ThunkAction<Promise<void>, StateType, unknown, AllTypes>;
type ThunkType = ThunkAction<void, StateType, unknown, AllTypes>;

export const getMainGraph = (uid: string): ThunkTypeAsync => {
  return async (dispatch) => {
    // console.log(`tra ta ta`, uid);
    dispatch(actionsSandbox.setMainGraph({ nodes: [], edges: [] }));
    try {
      let data = await sandboxAPI.getMainGraph(uid);
      console.log(`data`, data);
      dispatch(actionsSandbox.setMainGraph(data));
    } catch (err) {
      return Promise.reject(err);
    }
  };
};

export const getHistoryGraph = (uid: string): ThunkTypeAsync => {
  return async (dispatch) => {
    try {
      let data = await sandboxAPI.getHistoryGraph(uid);
      dispatch(actionsSandbox.setHistoryGraph(data));
    } catch (err) {
      return Promise.reject(err);
    }
  };
};

export const editMainGraph = (
  mainGraph: IMainGraph,
  nodes: NodeDataType[],
  edges: EdgeDataType[]
): ThunkType => {
  return (dispatch) => {
    let nodesEdit = mainGraph.nodes.filter((item) => item.id !== nodes[0].id);
    let edgesEdit = mainGraph.edges.filter(
      (item) => item.source !== nodes[0].id && item.target !== nodes[0].id
    );

    dispatch(
      actionsSandbox.setMainGraph({ nodes: nodesEdit, edges: edgesEdit })
    );
    dispatch(actionsSandbox.addNodeMainGraph({ nodes: nodes, edges: edges }));
  };
};

export default sandbox_Egor;
