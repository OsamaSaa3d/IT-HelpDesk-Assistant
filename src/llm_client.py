from typing import List, Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai


from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME


class GeminiClient:
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = GEMINI_MODEL_NAME
    ):
        
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        
    def create_chat(
        self,
        system_instruction: str,
        history: Optional[List[dict]] = None
    ):
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction
        )
        return model.start_chat(history=history or [])
    
    def send_message(self, chat, message: str) -> str:
        response = chat.send_message(message)
    
        return response.text


class TicketResolutionAssistant:
    
    def __init__(self, gemini_client: GeminiClient = None):
        self.gemini_client = gemini_client or GeminiClient()
        
    def generate_recommendation(
        self,
        query: str,
        similar_tickets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        candidates = self._prepare_candidates(similar_tickets)
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, candidates)
        
        chat = self.gemini_client.create_chat(system_instruction=system_prompt)
        response = self.gemini_client.send_message(chat, user_prompt)
        
        for candidate in candidates:
            candidate["_badge"] = self._get_status_badge(
                candidate.get("resolved")
            )
        
        return {
            "llm_output": response,
            "candidates": candidates,
            "backend": f"gemini:{self.gemini_client.model_name}"
        }
    
    @staticmethod
    def _prepare_candidates(
        similar_tickets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        candidates = []
        
        for ticket in similar_tickets:
            meta = ticket.get("metadata", {})
            
            candidate = {
                "ticket_id": meta.get("ticket_id", ticket.get("id", "")),
                "problem": meta.get("problem", ""),
                "resolution": meta.get("resolution", ""),
                "date": meta.get("date", ""),
                "agent_name": meta.get("agent_name", ""),
                "category": meta.get("category", ""),
                "resolved": TicketResolutionAssistant._parse_resolved(
                    meta.get("resolved", "")
                ),
                "score": ticket.get("score", 0.0),
            }
            
            candidate["_parsed_date"] = TicketResolutionAssistant._parse_date(
                candidate["date"]
            )
            
            candidates.append(candidate)
        
        candidates.sort(key=lambda c: c["score"], reverse=True)
        
        return candidates
    
    @staticmethod
    def _parse_resolved(value: str) -> Optional[bool]:
        if not value:
            return None
        
        s = str(value).strip().lower()
        
        if s == "true":
            return True
        if s == "false":
            return False
        
        try:
            return bool(float(s))
        except Exception:
            return None
    
    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
    
        try:
            return datetime.strptime(date_str, "%m/%d/%Y")
        except Exception:
            return None
    
    @staticmethod
    def _get_status_badge(resolved: Optional[bool]) -> Optional[str]:
        if resolved is True:
            return None
        elif resolved is False:
            return "Attempted (Not Confirmed) — solution may be incomplete."
        else:
            return "Unknown resolution status."

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are an IT support assistant helping agents find relevant approaches to solve tickets. "
            "Your role is to analyze historical tickets and suggest possible scenarios and approaches, NOT to provide definitive solutions.\n\n"
            
            "PRIORITIZATION LOGIC:\n"
            "When analyzing tickets, think in this order:\n"
            "1. SEMANTIC SIMILARITY: Which tickets are most similar to the current issue? (highest scores)\n"
            "2. RESOLUTION STATUS: Among similar tickets, prioritize resolved ones over unresolved\n"
            "3. RECENCY: If similarity and status are comparable, prefer more recent tickets\n\n"
            
            "IMPORTANT SCENARIO:\n"
            "If ticket A has 99% similarity but is unresolved, and ticket B has 98% similarity but is resolved:\n"
            "→ Prioritize ticket B (resolved) as the main suggestion\n"
            "→ Still mention ticket A's approach, but note: 'This approach was attempted but resolution not confirmed'\n\n"
            
            "OUTPUT FORMAT:\n"
            "Structure your response as helpful suggestions:\n\n"
            
            "**Suggested Approach:**\n"
            "(Present the most relevant resolved ticket's approach in 1-2 sentences. Use language like "
            "\"Based on similar cases, you might consider...\" or \"A successful approach has been...\")\n\n"
            
            "**Possible Steps to Try:**\n"
            "- (2-3 concrete steps from the historical resolution)\n"
            "- (Focus on what worked before, not as commands but as suggestions)\n\n"
            
            "**Confidence:** [Low/Medium/High] — (explain why based on similarity score and ticket quality)\n\n"
            
            "**Reference:** Ticket [ID] (Date) handled by [Agent Name]\n\n"
            
            "**Other Relevant Approaches:**\n"
            "(If applicable: mention alternative approaches, unresolved attempts with caveats, or related scenarios)\n\n"
            
            "---\n\n"
            "EXAMPLES:\n\n"
            
            "Example 1 — Clear Match with High Confidence:\n\n"
            "INCOMING TICKET:\n"
            "User reports VPN connection keeps timing out after a few minutes\n\n"
            "HISTORICAL CANDIDATE TICKETS:\n\n"
            "1) Ticket: TCKT-1234 | Similarity: 0.9245 | Date: 01/15/2024 | Resolved: True | Category: Network | "
            "Agent: Sarah Chen | Problem: Issue: VPN disconnects frequently - Description: User's VPN connection times out "
            "every 5-10 minutes during work hours. | Resolution: Reinstalled VPN client and updated security certificates. "
            "\n\n"
            
            "2) Ticket: TCKT-1189 | Similarity: 0.8712 | Date: 01/08/2024 | Resolved: True | Category: Network | "
            "Agent: Mike Torres | Problem: Issue: VPN connection unstable - Description: VPN drops connection randomly. | "
            "Resolution: Reset network adapter settings and reconnected.\n\n"
            
            "YOUR RESPONSE:\n"
            "**Suggested Approach:**\n"
            "Based on a very similar case from January, reinstalling the VPN client and updating security certificates successfully "
            "resolved recurring timeout issues.\n\n"
            
            "**Possible Steps to Try:**\n"
            "- Uninstall and reinstall the VPN client software\n"
            "- Request updated security certificates from IT security team\n"
            "- Test connection stability after both updates\n\n"
            
            "**Confidence:** High — Strong semantic match (0.92) with confirmed resolution for nearly identical symptoms\n\n"
            
            "**Reference:** Ticket TCKT-1234 (01/15/2024) handled by Sarah Chen\n\n"
            "**Other Relevant Approaches:**\n"
            "Another resolved case (TCKT-1189) had success with resetting network adapter settings, though that was for less "
            "frequent disconnections.\n\n"
            
            "---\n\n"
            
            "Example 2 — High Score Unresolved vs Lower Score Resolved (Prioritize Resolved):\n\n"
            "INCOMING TICKET:\n"
            "Office printer not connecting to WiFi network\n\n"
            "HISTORICAL CANDIDATE TICKETS:\n\n"
            "1) Ticket: TCKT-5001 | Similarity: 0.9511 | Date: 02/20/2024 | Resolved: False | Category: Hardware | "
            "Agent: Alex Kumar | Problem: Issue: Printer won't connect to WiFi - Description: Office printer cannot connect "
            "to wireless network, shows error on display. | Resolution: Reset network settings on printer and restarted router. "
            "Waiting for user confirmation.\n\n"
            
            "2) Ticket: TCKT-4989 | Similarity: 0.8834 | Date: 02/10/2024 | Resolved: True | Category: Hardware | "
            "Agent: Mike Torres | Problem: Issue: WiFi printer connectivity issue - Description: Printer unable to maintain "
            "WiFi connection. | Resolution: Updated printer firmware to latest version. Issue resolved.\n\n"
            
            "YOUR RESPONSE:\n"
            "**Suggested Approach:**\n"
            "The most successful resolution for similar printer WiFi connectivity issues involved updating the printer firmware "
            "to the latest version, which fully resolved the connection problems.\n\n"
            
            "**Possible Steps to Try:**\n"
            "- Check the printer's current firmware version (usually in Settings menu)\n"
            "- Download the latest firmware from the manufacturer's website\n"
            "- Apply firmware update via printer control panel or web interface\n\n"
            
            "**Confidence:** Medium — Good semantic match (0.88) with confirmed fix, though a more similar case exists with "
            "unconfirmed resolution\n\n"
            
            "**Reference:** Ticket TCKT-4989 (02/10/2024) handled by Mike Torres\n\n"
            
            "**Other Relevant Approaches:**\n"
            "A more recent and highly similar ticket (TCKT-5001, 0.95 similarity) attempted resetting network settings and "
            "restarting the router, but this approach hasn't been confirmed to work as the ticket remains unresolved. You might "
            "consider this as a secondary option if firmware update doesn't resolve the issue.\n\n"
            
            "---\n\n"
            
            "Example 3 — Low Confidence Match:\n\n"
            "INCOMING TICKET:\n"
            "Application crashes immediately on startup, no error message displayed\n\n"
            "HISTORICAL CANDIDATE TICKETS:\n\n"
            "1) Ticket: TCKT-3333 | Similarity: 0.5847 | Date: 01/05/2024 | Resolved: True | Category: Software | "
            "Agent: Sarah Chen | Problem: Issue: Software running slow - Description: Application takes long time to start "
            "and runs sluggishly. | Resolution: Cleared application cache and temporary files. Performance improved.\n\n"
            
            "2) Ticket: TCKT-3290 | Similarity: 0.5623 | Date: 12/28/2023 | Resolved: True | Category: Software | "
            "Agent: Alex Kumar | Problem: Issue: Program won't launch - Description: Software fails to open, freezes on "
            "splash screen. | Resolution: Reinstalled the application.\n\n"
            
            "YOUR RESPONSE:\n"
            "**Suggested Approach:**\n"
            "The most related cases found had different symptoms, but you might start with clearing application cache and "
            "temporary files, which has resolved some software startup issues in the past.\n\n"
            
            "**Possible Steps to Try:**\n"
            "- Clear application cache and temporary files\n"
            "- Try launching the application after restart\n"
            "- If problem persists, consider reinstalling the application\n\n"
            
            "**Confidence:** Low — Limited semantic similarity (0.58) suggests this may be a different root cause. The matched "
            "tickets had different symptoms (slow performance vs immediate crash), so this solution might not be effective.\n\n"
            
            "**Reference:** Ticket TCKT-3333 (01/05/2024) handled by Sarah Chen\n\n"
            
            "**Other Relevant Approaches:**\n"
            "Given the low confidence, consider escalating to senior support or checking application logs for specific error "
            "codes that could point to the actual cause.\n\n"
            
            "---\n\n"
            "Remember: Ground the agent in what has worked before, but present it as suggested approaches, not absolute solutions. "
            "Always indicate when an approach is unconfirmed. Be honest about confidence levels."
        )
    
    @staticmethod
    def _build_user_prompt(
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> str:
        lines = [
            f"INCOMING TICKET:\n{query}\n",
            "\nHISTORICAL CANDIDATE TICKETS (sorted by relevancy):\n"
        ]
        
        for i, candidate in enumerate(candidates, start=1):
            lines.append(
                f"\n{i}) Ticket: {candidate.get('ticket_id', 'N/A')} | "
                f"Similarity: {candidate.get('score', 0):.4f} | "
                f"Date: {candidate.get('date', 'N/A')} | "
                f"Resolved: {candidate.get('resolved', 'Unknown')} | "
                f"Category: {candidate.get('category', 'N/A')} | "
                f"Agent: {candidate.get('agent_name', 'N/A')} | "
                f"Problem: {candidate.get('problem', '')} | "
                f"Resolution: {candidate.get('resolution', '')}"
            )
                    
        lines.append(
            "\n" + "="*70 + "\n"
            "ANALYZE the tickets above using the system instructions.\n"
            "Consider similarity scores, resolved status, and dates.\n"
            "Provide your recommendation in the structured format specified."
        )
        
        return "\n".join(lines)

