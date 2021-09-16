import { instance } from "../baseURL";
import {
  EdgeDataType,
  NodeDataType,
} from "../../workSpace/pages/sandbox/GraphEditor/GraphEditorDirectedGraph/GraphEditorDirectedGraph";
import { IAdd, IValidate } from "./pipelineInterface";

export const pipelineAPI = {
  async postValidate(
    uid: string,
    nodes: NodeDataType[],
    edges: EdgeDataType[]
  ) {
    try {
      const res = await instance.post<IValidate>(`/pipelines/validate`, {
        uid: uid,
        nodes: nodes,
        edges: edges,
      });
      return res.data;
    } catch (error: any) {
      console.error(`error`, error.response);
      return Promise.reject(error);
    }
  },
  async postAdd(uid: string, nodes: NodeDataType[], edges: EdgeDataType[]) {
    try {
      const res = await instance.post<IAdd>(`/pipelines/add`, {
        uid: uid,
        nodes: nodes,
        edges: edges,
      });
      return res.data;
    } catch (error: any) {
      console.error(`error`, error.response);
      return Promise.reject(error);
    }
  },
};
