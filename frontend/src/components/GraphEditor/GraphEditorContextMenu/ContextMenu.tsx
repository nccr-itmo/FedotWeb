import React, { FC } from "react";
import style from "./contextMenu.module.scss";
import { offsetContextMenuType } from "../GraphEditorDirectedGraph/GraphEditorDirectedGraph";

export interface IContextMenu {
  firstName: string;
  secondName?: string;
  thirdName?: string;
  fourthName?: string;
  offset: offsetContextMenuType | undefined;
  firstAction(): void;
  secondAction?: () => void;
  thirdAction?: () => void;
  fourthAction?: () => void;
}

const ContextMenu: FC<IContextMenu> = (props) => {
  let position: any;

  if (props.offset) {
    position = {
      position: "absolute",
      top: `${props.offset.y}px`,
      left: `${props.offset.x}px`,
      zIndex: 3,
    };
  }

  return (
    <div className={style.root} style={position}>
      <button onClick={props.firstAction}>{props.firstName}</button>
      {props.secondName && (
        <button onClick={props.secondAction}>{props.secondName}</button>
      )}
      {props.thirdName && (
        <button onClick={props.thirdAction}>{props.thirdName}</button>
      )}
      {props.fourthName && (
        <button onClick={props.fourthAction}>{props.fourthName}</button>
      )}
    </div>
  );
};
export default ContextMenu;
