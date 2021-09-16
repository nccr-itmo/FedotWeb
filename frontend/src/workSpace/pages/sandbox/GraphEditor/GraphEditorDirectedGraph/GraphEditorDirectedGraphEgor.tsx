import React, { useEffect, useRef, useState } from "react";
import { runDirectedGraph } from "./directedGraphGenerator";
import styles from "./directedGraph.module.scss";
import Loader from "react-loader-spinner";
import ContextMenu from "../GraphEditorContextMenu/ContextMenu";
import { ClickAwayListener } from "@material-ui/core";
import { useDispatch, useSelector } from "react-redux";
import GraphEditorModal from "../GraphEditorModal/GraphEditorModal";
import useMousePosition from "../../../../../hooks/useMousePosition";
import { StateType } from "../../../../../redux/store";
import { actionsSandbox } from "../../../../../redux/reducers/sandBox/sandbox-reducer";

export type EdgeDataType = {
  source: number;
  target: number;
};
export type NodeDataType = {
  id: number;
  display_name: string;
  model_name: string;
  class: string;
  params: any;
  parents: number[];
  children: number[];
};

type DirectedGraphProps = {
  edgesData: EdgeDataType[];
  nodesData: NodeDataType[];
};

export type offsetContextMenuType = {
  x: number;
  y: number;
};

export type dataContextType = {
  data: number;
  offset: offsetContextMenuType;
};
export type edgeValueType = {
  v: string;
  w: string;
};
export type edgeContextType = {
  data: edgeValueType;
  offset: offsetContextMenuType;
};
const GraphEditorDirectedGraph = ({
  edgesData,
  nodesData,
}: DirectedGraphProps) => {
  const dispatch = useDispatch();
  const containerRef = useRef(null);
  const position = useMousePosition();
  const [showLoader, setShowLoader] = useState(true);
  const [dataContext, setDataContext] = useState<dataContextType | undefined>(
    undefined
  );
  const [edgeContext, setEdgeContext] = useState<any>();
  const [linePath, setLinePath] = useState<dataContextType | null>(null);
  const { mainGraph } = useSelector((state: StateType) => state.sandbox_Egor);

  useEffect((): any => {
    if (edgesData && nodesData) {
      if (edgesData.length === 0 || nodesData.length === 0) return;
      let destroyFn;
      let savedData: dataContextType | null;

      const handleContextMenuNode = (e: any, d: any) => {
        console.log(`d`, d);
        e.preventDefault();
        setDataContext({
          data: d,
          offset: {
            x: e.offsetX,
            y: e.offsetY,
          },
        });
      };

      const handleContextMenuEdge = (e: any, d: any) => {
        e.preventDefault();
        setEdgeContext({
          data: d,
          offset: {
            x: e.offsetX,
            y: e.offsetY,
          },
        });
      };

      const handleMouseDownNode = (d: dataContextType) => {
        console.log(`down`, d);
        setLinePath(d);
        savedData = d;
      };
      const handleMouseUpNode = (d: dataContextType) => {
        console.log(`up`, d);
        if (d && savedData) {
          dispatch(
            actionsSandbox.addEdgeMainGraph({
              source: savedData.data,
              target: d.data,
            })
          );
        }
        savedData = null;
        setLinePath(null);
      };

      const handleMouseUpSvg = (e: any, d: any) => {
        savedData = null;
        setLinePath(null);
      };

      setShowLoader(false);

      // отрисовка графа
      if (containerRef.current) {
        const { destroy } = runDirectedGraph(
          containerRef.current,
          edgesData,
          nodesData,
          handleContextMenuNode
          // handleContextMenuEdge
          // handleMouseDownNode,
          // handleMouseUpNode,
          // handleMouseUpSvg
        );
        destroyFn = destroy;
      }

      return destroyFn;
    }
  }, [nodesData, edgesData, dispatch]);

  const handleClickAwayNodeContext = () => {
    setDataContext(undefined);
  };
  const handleClickAwayEdgeContext = () => {
    setEdgeContext(undefined);
  };

  const handleAddNode = (): void => {
    if (dataContext) {
      //TODO: необходимо создавать уникальный id
      const newNode = {
        ...mainGraph.nodes[0],
        id: 1000 + mainGraph.nodes.length,
      };
      const newEdge = { source: newNode.id, target: Number(dataContext.data) };
      dispatch(
        actionsSandbox.addNodeMainGraph({
          nodes: [newNode],
          edges: [newEdge],
        })
      );
      setDataContext(undefined);
    }
  };

  const handleSecondActionNode = () => {
    dispatch(actionsSandbox.toggleEditModal());
  };

  const handleDeleteEdge = () => {
    dispatch(actionsSandbox.deleteEdgeMainGraph(edgeContext.data));
    setEdgeContext(undefined);
  };

  useEffect(() => {
    // console.log(`### linePath`, linePath);
  }, [linePath]);

  return (
    <div ref={containerRef} className={styles.container}>
      {showLoader && (
        <Loader
          type="MutatingDots"
          color="#263238"
          secondaryColor="#0199E4"
          height={100}
          width={100}
        />
      )}
      <ClickAwayListener onClickAway={handleClickAwayNodeContext}>
        <div>
          {/* {dataContext && (
            <ContextMenu
              firstName={"add node"}
              secondName={"edit"}
              offset={dataContext?.offset}
              firstAction={handleAddNode}
              secondAction={handleSecondActionNode}
            />
          )} */}
        </div>
      </ClickAwayListener>
      <ClickAwayListener onClickAway={handleClickAwayEdgeContext}>
        <div>
          {/* {edgeContext && (
            <ContextMenu
              firstName={"delete edge"}
              offset={edgeContext?.offset}
              firstAction={handleDeleteEdge}
            />
          )} */}
        </div>
      </ClickAwayListener>
      <GraphEditorModal dataContext={dataContext} />
      {/* {linePath && position.x && position.y && (
        <svg
          width={"100%"}
          height={"100%"}
          style={{ position: "absolute", zIndex: 1 }}
        >
          <line
            x1={linePath.offset.x}
            y1={linePath.offset.y}
            x2={position.x}
            y2={position.y}
            stroke="#2F80ED"
            strokeWidth={3}
            markerEnd="url(#arrowhead)"
          />
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="0"
              refY="3.5"
              orient="auto"
            >
              <polygon fill="#2F80ED" points="0 0, 10 3.5, 0 7" />
            </marker>
          </defs>
        </svg>
      )} */}
    </div>
  );
};

export default GraphEditorDirectedGraph;
