import { instance } from "./baseURL";

import {
  HistoryEdgeType,
  HistoryNodeIndividualType,
  HistoryNodeOperatorType,
} from "../workSpace/pages/history/HistoryGraph/HistoryGraph";
import {
  EdgeDataType,
  NodeDataType,
} from "../workSpace/pages/sandbox/GraphEditor/GraphEditorDirectedGraph/GraphEditorDirectedGraph";

export interface IMainGraph {
  nodes: NodeDataType[];
  edges: EdgeDataType[];
}

export interface IHistoryGraph {
  nodes: (HistoryNodeIndividualType | HistoryNodeOperatorType)[];
  edges: HistoryEdgeType[];
}

export interface IPipelineImage {
  image_url: string;
  uid: string;
}

export const sandboxAPI = {
  async getMainGraph(uid: string) {
    try {
      const res = await instance.get<IMainGraph>("/pipelines/" + uid);
      return res.data;
    } catch (err) {
      return Promise.reject(err);
    }
  },
  async getHistoryGraph(uid: string) {
    try {
      const res = await instance.get<IHistoryGraph>("/composer/" + uid);
      return res.data;
    } catch (err) {
      return Promise.reject(err);
    }
  },
  async getChainsImage(uid: string) {
    try {
      const res = await instance.get<IPipelineImage>("/pipelines/image/" + uid);
      return res.data;
    } catch (err) {
      return Promise.reject(err);
    }
  },
};
