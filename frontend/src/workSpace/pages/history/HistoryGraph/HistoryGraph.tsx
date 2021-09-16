import React, { useEffect, useState } from "react";
import { runHistory } from "./historyGraphGenerator";
import styles from "./HistoryGraph.module.scss";
import Loader from "react-loader-spinner";

export type HistoryEdgeType = {
  source: string;
  target: string;
};
export type HistoryNodeIndividualType = {
  uid: string;
  type: string;
  pipeline_id: any;
  full_name?: any;
  objs?: any;
  gen_id: number;
  ind_id: number;
};

export type HistoryNodeOperatorType = {
  uid: string;
  type: string;
  full_name?: any;
  prev_gen_id: number;
  next_gen_id: number;
  operator_id: number;
  name: string[];
};

type DirectedGraphProps = {
  edgesData: HistoryEdgeType[];
  nodesData: (HistoryNodeIndividualType | HistoryNodeOperatorType)[];
  onClick(d: any): void;
  nodeHoverTooltip(d: any): void;
};

export type dataContextType = {
  data: number;
};

const HistoryGraph = ({
  edgesData,
  nodesData,
  onClick,
  nodeHoverTooltip,
}: DirectedGraphProps) => {
  const containerRef = React.useRef(null);
  const [showLoader, setShowLoader] = useState(true);

  useEffect((): any => {
    if (edgesData.length === 0 || nodesData.length === 0) return;
    let destroyFn;

    if (containerRef.current) {
      const { destroy } = runHistory(
        containerRef.current,
        edgesData,
        nodesData,
        nodeHoverTooltip,
        onClick
      );
      destroyFn = destroy;
    }

    setShowLoader(false);
    return destroyFn;
  }, [nodesData, edgesData, onClick, nodeHoverTooltip]);

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
    </div>
  );
};

export default HistoryGraph;
