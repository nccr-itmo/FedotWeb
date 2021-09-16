import React, { useEffect, useRef, useState } from "react";
import { runDirectedGraph } from "./directedGraphGenerator";
import styles from "./directedGraph.module.scss";
import Loader from "react-loader-spinner";
import ContextMenu from "../GraphEditorContextMenu/ContextMenu";
import { ClickAwayListener } from "@material-ui/core";
import { useDispatch, useSelector } from "react-redux";
import GraphEditorModal from "../GraphEditorModal/GraphEditorModal";
import { StateType } from "../../../../../redux/store";
import { actionsSandbox } from "../../../../../redux/reducers/sandBox/sandbox-reducer";
import {
  setEditOrAddFormOpened,
  setInitialValuesEditForm,
} from "../../../../../redux/reducers/sandBox/sandBoxReducer";

export type EdgeDataType = {
  source: number;
  target: number;
};
export type NodeDataType = {
  id: number;
  display_name: string;
  model_name: string;
  type: string;
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
  const [showLoader, setShowLoader] = useState(true);
  const [dataContext, setDataContext] = useState<dataContextType | undefined>(
    undefined
  );
  const [edgeContext, setEdgeContext] = useState<any>();
  const { mainGraph } = useSelector((state: StateType) => state.sandbox_Egor);

  useEffect(() => {
    if (edgesData && nodesData) {
      if (edgesData.length === 0 || nodesData.length === 0) return;
      let destroyFn;

      const handleContextMenuNode = (e: any, id: any) => {
        // console.log(`d`, id);
        e.preventDefault();

        // console.log(`effff`, e);

        setDataContext({
          data: id,
          offset: {
            x: e.offsetX,
            y: e.clientY > 350 ? e.offsetY - 140 : e.offsetY,
          },
        });
      };

      setShowLoader(false);

      // отрисовка графа
      if (containerRef.current) {
        const { destroy } = runDirectedGraph(
          containerRef.current,
          edgesData,
          nodesData,
          handleContextMenuNode
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

  //логика добавления графа
  // const handleAddNode = (): void => {
  //   if (dataContext) {
  //     //TODO: необходимо создавать уникальный id
  //     const newNode = {
  //       ...mainGraph.nodes[0],
  //       id: 1000 + mainGraph.nodes.length,
  //     };
  //     const newEdge = { source: newNode.id, target: Number(dataContext.data) };
  //     dispatch(
  //       actionsSandbox.addNodeMainGraph({
  //         nodes: [newNode],
  //         edges: [newEdge],
  //       })
  //     );
  //     setDataContext(undefined);
  //   }
  // };

  const handleSecondActionNode = () => {
    dispatch(actionsSandbox.toggleEditModal());
  };

  const deleteNode = () => {
    if (dataContext)
      dispatch(actionsSandbox.deleteNodeMainGraph(dataContext.data));

    setDataContext(undefined);
  };

  const openEditNode = () => {
    // получаю ифо об узле, его родителей, его потомков
    if (dataContext) {
      let node = mainGraph.nodes.find((item) => +dataContext.data === item.id);
      dispatch(setInitialValuesEditForm(node));
    }
    // открываю form (timeout для обновления формы)
    setTimeout(() => {
      dispatch(setEditOrAddFormOpened({ isOpen: true, type: "edit" }));
    }, 300);

    // закрываю меню
    setDataContext(undefined);
  };

  // const handleDeleteEdge = () => {
  //   dispatch(actionsSandbox.deleteEdgeMainGraph(edgeContext.data));
  //   setEdgeContext(undefined);
  // };

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
      {/* контекстное меню узла  */}
      <ClickAwayListener onClickAway={handleClickAwayNodeContext}>
        <div>
          {dataContext && (
            <ContextMenu
              buttons={[
                { name: "edit", action: openEditNode },
                // { name: "params", action: handleSecondActionNode },
                { name: "delete", action: deleteNode },
              ]}
              offset={dataContext?.offset}
            />
          )}
        </div>
      </ClickAwayListener>
      <GraphEditorModal dataContext={dataContext} />
    </div>
  );
};

export default GraphEditorDirectedGraph;
