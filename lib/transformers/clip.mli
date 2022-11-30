module Tokenizer : sig
  type t =
    { re : Re2.t
    ; encoder : (string, int) Hashtbl.t
    ; decoder : (int, string) Hashtbl.t
    ; bpe_ranks : (string * string, int) Hashtbl.t
    ; start_of_text_token : int
    ; end_of_text_token : int
    }

  val make : string -> t
end
